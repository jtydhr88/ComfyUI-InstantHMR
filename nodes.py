import math
import os

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
import folder_paths
from comfy_api.latest import InputImpl, io
from comfy_extras.nodes_sam3d_body import _per_frame_bboxes_from_detections
from comfy_extras.sam3d_body.utils import compute_canonical_colors, compute_hand_vert_mask, inputs_from_sam3_track

from .instanthmr_model import InstantHMR, backbone_from_state_dict

InstantHMRModel = io.Custom("INSTANTHMR_MODEL")
SAM3TrackData = io.Custom("SAM3_TRACK_DATA")
MHRPoseData = io.Custom("MHR_POSE_DATA")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CROP_EXPAND = 1.2
RECORD_DIR = "instanthmr_live"
CHARACTER_EXTENSIONS = (".fbx", ".glb", ".gltf", ".vrm")
LIVE_CAMERAS = ["Phone / other device", "This device"]
MHR_BODY = "MHR body"
# Focal when the camera is unknown, as a multiple of the image diagonal (InstantHMR's training default).
FOCAL_DIAG = 1.05


class InstantHMRLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="InstantHMRLoader",
            display_name="Load InstantHMR Model",
            category="image/detection",
            inputs=[io.Combo.Input("model_file", options=folder_paths.get_filename_list("detection"))],
            outputs=[InstantHMRModel.Output(display_name="instanthmr_model")],
        )

    @classmethod
    def execute(cls, model_file) -> io.NodeOutput:
        sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("detection", model_file), safe_load=True)
        load_device = comfy.model_management.get_torch_device()
        dtype = comfy.model_management.unet_dtype(load_device, model_params=-1, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32],
                                                  weight_dtype=comfy.utils.weight_dtype(sd))
        manual_cast_dtype = comfy.model_management.unet_manual_cast(dtype, load_device)
        operations = comfy.ops.pick_operations(dtype, manual_cast_dtype, load_device=load_device, disable_fast_fp8=True)

        model = InstantHMR(backbone_from_state_dict(sd), dtype=dtype, operations=operations)
        model.load_state_dict(sd, strict=True)
        model.dtype = manual_cast_dtype or dtype
        patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=load_device, offload_device=comfy.model_management.unet_offload_device(),
                                                       size=comfy.model_management.module_size(model))
        return io.NodeOutput(patcher)


def crop_batch(frame, boxes, focal, size):
    """The square 1.2x box crops InstantHMR is trained on, their CLIFF vectors, and the crop squares (x0, y0, side)."""
    _, H, W = frame.shape
    # float64 like the reference: in float32 a box edge on an integer pixel can floor one pixel off.
    x1, y1, x2, y2 = boxes.double().unbind(-1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    box = torch.maximum(x2 - x1, y2 - y1)
    cliff = torch.stack([torch.atan((cx - W / 2) / focal), torch.atan((cy - H / 2) / focal), box / focal], -1)

    side = box * CROP_EXPAND
    sx, sy = cx - side / 2, cy - side / 2
    # The reference cuts the pixel-aligned patch around the square and resizes it to the crop size.
    px, py = torch.floor(sx), torch.floor(sy)
    pw, ph = torch.ceil(sx + side) - px, torch.ceil(sy + side) - py
    t = (torch.arange(size, device=frame.device, dtype=torch.float64) + 0.5) / size
    gx = (px[:, None] + t[None] * pw[:, None]) * (2 / W) - 1
    gy = (py[:, None] + t[None] * ph[:, None]) * (2 / H) - 1
    grid = torch.stack([gx[:, None, :].expand(-1, size, -1), gy[:, :, None].expand(-1, -1, size)], -1).float()
    crops = F.grid_sample(frame[None].expand(len(boxes), -1, -1, -1), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=frame.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=frame.device).view(1, 3, 1, 1)
    return (crops - mean) / std, cliff.float(), torch.stack([sx, sy, side], -1).float()


def drop_absent_tracks(boxes, masks):
    """Leave out track slots whose mask is empty in a frame, which inputs_from_sam3_track boxes as the full frame.
    Person indices are track ids, so a gap below the last present slot keeps the slot's nearest box."""
    present = torch.stack([m.flatten(1).any(1) for m in masks])
    keep = present.any(0)
    present = present[:, keep]
    boxes = [b[keep] for b in boxes]
    masks = [m[keep] for m in masks]
    frames = torch.arange(len(boxes))
    for f in range(len(boxes)):
        count = int(present[f].nonzero().max()) + 1 if present[f].any() else 0
        for k in range(count):
            if not present[f, k]:
                seen = frames[present[:, k]]
                boxes[f][k] = boxes[int(seen[(seen - f).abs().argmin()])][k]
        boxes[f], masks[f] = boxes[f][:count], masks[f][:count]
    return boxes, masks


class InstantHMRPredict(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="InstantHMRPredict",
            display_name="Run InstantHMR Prediction",
            category="image/detection",
            inputs=[
                InstantHMRModel.Input("instanthmr_model"),
                io.Image.Input("image"),
                SAM3TrackData.Input("track_data", optional=True, tooltip="Person tracks from SAM3 Video Track, for multi-person video"),
                io.BoundingBox.Input("bboxes", optional=True, force_input=True, tooltip="Per-frame person boxes, e.g. from RT-DETR Detect with class 'person'"),
                io.Float.Input("fov", default=0.0, min=0.0, max=180.0, step=0.1, advanced=True,
                               tooltip="Vertical FoV in degrees. Sets the depth and absolute scale. 0 = focal of 1.05x the image diagonal."),
                io.Int.Input("batch_size", default=64, min=1, max=1024, advanced=True, tooltip="Person crops per forward pass"),
            ],
            outputs=[MHRPoseData.Output("mhr_pose_data")],
        )

    @classmethod
    def execute(cls, instanthmr_model, image, track_data=None, bboxes=None, fov=0.0, batch_size=64) -> io.NodeOutput:
        B, H, W, _ = image.shape
        per_frame_boxes, per_frame_masks = None, None
        if track_data is not None:
            per_frame_boxes, per_frame_masks = inputs_from_sam3_track(track_data, B, H, W)
            if per_frame_boxes is not None:
                per_frame_boxes, per_frame_masks = drop_absent_tracks(per_frame_boxes, per_frame_masks)
        if per_frame_boxes is None and bboxes:
            per_frame_boxes = _per_frame_bboxes_from_detections(bboxes, B)
        if per_frame_boxes is None:
            per_frame_boxes = [torch.tensor([[0.0, 0.0, float(W), float(H)]]) for _ in range(B)]
        focal = H / (2 * math.tan(math.radians(fov) / 2)) if fov > 0 else FOCAL_DIAG * math.hypot(H, W)

        comfy.model_management.load_models_gpu([instanthmr_model], memory_required=instanthmr_model.model.memory_used_forward(batch_size))
        model = instanthmr_model.model
        device = instanthmr_model.load_device

        jobs = [(f, k) for f in range(B) for k in range(per_frame_boxes[f].shape[0])]
        frames_out = [[] for _ in range(B)]
        pbar = comfy.utils.ProgressBar(len(jobs))
        for start in range(0, len(jobs), batch_size):
            chunk = jobs[start:start + batch_size]
            crops, cliffs, squares = [], [], []
            for f in sorted({f for f, _ in chunk}):
                ks = [k for g, k in chunk if g == f]
                frame = image[f].to(device=device, dtype=torch.float32).permute(2, 0, 1)
                c, cl, sq = crop_batch(frame, per_frame_boxes[f][ks].to(device), focal, model.image_size)
                crops.append(c)
                cliffs.append(cl)
                squares.append(sq)
            squares = torch.cat(squares)
            out = model(torch.cat(crops).to(model.dtype), torch.cat(cliffs).to(model.dtype))
            out["pred_keypoints_2d"] = (out.pop("pred_keypoints_2d_crop") + 1) * 0.5 * squares[:, None, 2:] + squares[:, None, :2]
            out = {k: v.float().cpu().numpy() for k, v in out.items()}

            for i, (f, k) in enumerate(chunk):
                masks = per_frame_masks[f] if per_frame_masks is not None else None
                frames_out[f].append({
                    "bbox": per_frame_boxes[f][k].numpy(),
                    "focal_length": np.float32(focal),
                    "pred_keypoints_3d": out["pred_keypoints_3d"][i],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][i],
                    "pred_vertices": out["pred_vertices"][i],
                    "pred_cam_t": out["pred_cam_t"][i],
                    "global_rot": out["global_rot"][i],
                    "body_pose_params": out["body_pose"][i],
                    "hand_pose_params": out["hand"][i],
                    "scale_params": out["scale"][i],
                    "shape_params": out["shape"][i],
                    "expr_params": out["face"][i],
                    "mask": (masks[k] if masks.shape[0] > 1 else masks[0]) if masks is not None else None,
                    "pred_joint_coords": out["pred_joint_coords"][i],
                    "pred_global_rots": out["joint_global_rots"][i],
                    "mhr_model_params": out["mhr_model_params"][i],
                    "pred_face_keypoints_3d": None,
                    "pred_face_keypoints_2d": None,
                })
            pbar.update(len(chunk))

        return io.NodeOutput({
            "frames": frames_out,
            "faces": model.head_pose.faces_np(),
            "image_size": (int(H), int(W)),
            "canonical_colors": compute_canonical_colors(model),
            "hand_vert_mask": compute_hand_vert_mask(model),
        })


def recording_files():
    root = os.path.join(folder_paths.get_input_directory(), RECORD_DIR)
    if not os.path.isdir(root):
        return []
    return sorted((f"{RECORD_DIR}/{f}" for f in os.listdir(root) if f.endswith(".mp4")), reverse=True)


def character_files():
    root = folder_paths.get_input_directory()
    files = []
    for folder, _, names in os.walk(root):
        files += [os.path.relpath(os.path.join(folder, n), root).replace(os.sep, "/") for n in names if n.lower().endswith(CHARACTER_EXTENSIONS)]
    return sorted(files)


class InstantHMRLiveCapture(io.ComfyNode):
    """Live preview from this device's camera or a phone on the LAN; the recorded clip is the output.
    Like WebcamCapture, the live settings are inputs the node's frontend reads; execute only loads the recording."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="InstantHMRLiveCapture",
            display_name="InstantHMR Live Capture",
            category="image/detection",
            inputs=[
                io.Combo.Input("recording", options=recording_files(), tooltip="A clip recorded with the live view"),
                io.Combo.Input("model", options=[f for f in folder_paths.get_filename_list("detection") if "instanthmr" in f.lower()],
                               tooltip="InstantHMR model for the live view"),
                io.Combo.Input("detector", options=[f for f in folder_paths.get_filename_list("diffusion_models") if "rt_detr" in f.lower()],
                               tooltip="RT-DETR person detector for the live view"),
                io.Combo.Input("camera", options=LIVE_CAMERAS, tooltip="Phone / other device shows a QR code to open on the phone"),
                io.Combo.Input("character", options=[MHR_BODY, *character_files()],
                               tooltip="Body in the 3D view: the MHR mesh, or a rigged FBX, glTF or VRM character from the input folder"),
                io.String.Input("vmc", default="", placeholder="host:port, e.g. 127.0.0.1:39539",
                                tooltip="Stream the first person to a VMC protocol app (VSeeFace, Warudo, VNyan...); empty = off"),
                io.Float.Input("fov", default=0.0, min=0.0, max=179.0, step=1.0, advanced=True,
                               tooltip="Vertical field of view in degrees; 0 uses a focal of 1.05x the image diagonal"),
                io.Int.Input("detect_every", default=2, min=1, max=10, advanced=True,
                             tooltip="Run the detector every N frames and reuse its boxes in between"),
            ],
            outputs=[io.Video.Output()],
        )

    @classmethod
    def execute(cls, recording, **live_settings) -> io.NodeOutput:
        return io.NodeOutput(InputImpl.VideoFromFile(folder_paths.get_annotated_filepath(recording)))

    @classmethod
    def validate_inputs(cls, recording, model, detector, character):
        # Only the recording is needed to run; the live models and character may be missing on this machine.
        if recording not in recording_files():
            return f"Recording not found: {recording}"
        return True
