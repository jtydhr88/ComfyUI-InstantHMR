import asyncio
import json
import math
import os
import socket
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO

import av
import numpy as np
import torch
from aiohttp import web
from PIL import Image

import comfy.model_management
import comfy.sd
import folder_paths
import comfy.utils
from comfy.ldm.rt_detr.rtdetr_v4 import COCO_CLASSES

from .humanoid import BONES, HumanoidRig, VMCSender, mat_to_quat
from .nodes import FOCAL_DIAG, RECORD_DIR, InstantHMRLoader, crop_batch, recording_files

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_PERSONS = 4
PERSON = COCO_CLASSES.index("person")

# One worker so live rooms take turns on the GPU instead of racing each other's model loads.
GPU = ThreadPoolExecutor(max_workers=1, thread_name_prefix="instanthmr-live")


def pack(header, *buffers):
    header["sizes"] = [len(b) for b in buffers]
    head = json.dumps(header).encode()
    return struct.pack("<I", len(head)) + head + b"".join(buffers)


class Room:
    def __init__(self):
        self.viewers = set()
        self.cameras = set()
        self.sending = set()
        self.settings = {"model": None, "detector": None, "fov": 0.0, "detect_every": 2, "vmc": ""}
        self.boxes = torch.zeros(0, 4)
        self.frame_index = 0
        self.recording = None
        self.vmc = None

    def set_vmc(self, target):
        if self.vmc:
            self.vmc.close()
        self.vmc = VMCSender(target) if target else None

    def model(self, kind, name):
        key = (kind, name)
        if key not in models:
            if kind == "hmr":
                models[key] = InstantHMRLoader.execute(name).args[0]
            elif kind == "humanoid":
                models[key] = HumanoidRig(self.model("hmr", name).model.mhr)
            else:
                models[key] = comfy.sd.load_diffusion_model(folder_paths.get_full_path_or_raise("diffusion_models", name))
        return models[key]

    def infer(self, jpeg):
        s = self.settings
        with torch.inference_mode():
            hmr, detector = self.model("hmr", s["model"]), self.model("detector", s["detector"])
            # Loading re-stages the weights, which costs more than a frame; only reload after a workflow evicted them.
            loaded = comfy.model_management.loaded_models()
            if hmr not in loaded or detector not in loaded:
                comfy.model_management.load_models_gpu([hmr, detector])
            device = hmr.load_device
            frame = torch.from_numpy(np.array(Image.open(BytesIO(jpeg)).convert("RGB"))).to(device)
            H, W = frame.shape[:2]
            image = frame.permute(2, 0, 1).float() / 255
            t0 = time.perf_counter()
            if self.frame_index % max(1, int(s["detect_every"])) == 0:
                det = detector.model.diffusion_model(comfy.utils.common_upscale(image[None], 640, 640, "bilinear", crop="disabled"), (W, H))[0]
                keep = (det["scores"] > 0.5) & (det["labels"] == PERSON)
                order = det["scores"][keep].argsort(descending=True)[:MAX_PERSONS]
                self.boxes = det["boxes"][keep][order].float()
            self.frame_index += 1
            t1 = time.perf_counter()

            persons, verts = [], b""
            if len(self.boxes):
                model = hmr.model
                fov = float(s["fov"])
                focal = H / (2 * math.tan(math.radians(fov) / 2)) if fov > 0 else FOCAL_DIAG * math.hypot(H, W)
                crops, cliff, squares = crop_batch(image, self.boxes.to(device), focal, model.image_size)
                out = model(crops.to(model.dtype), cliff.to(model.dtype))
                kp2d = (out["pred_keypoints_2d_crop"] + 1) * 0.5 * squares[:, None, 2:] + squares[:, None, :2]
                cam_t = out["pred_cam_t"]
                verts = (out["pred_vertices"] + cam_t[:, None]).float().cpu().numpy().tobytes()
                kp2d, cam_t = kp2d.float().cpu().numpy(), cam_t.cpu().numpy()
                # Joint coords are in the camera frame (Y down); the bone rotations and the viewer use the MHR frame (Y up).
                joints = out["pred_joint_coords"].float().cpu().numpy() * [1, -1, -1]
                hips = joints[:, 1] + cam_t * [1, -1, -1]
                legs = np.linalg.norm(joints[:, 2] - joints[:, 3], axis=1) + np.linalg.norm(joints[:, 3] - joints[:, 4], axis=1)
                humanoid = self.model("humanoid", s["model"])
                world = humanoid.world(out["joint_global_rots"])
                bones = mat_to_quat(world).astype(np.float32).round(4)
                persons = [{"bbox": self.boxes[i].tolist(), "kp2d": kp2d[i].round(1).ravel().tolist(), "cam_t": cam_t[i].tolist(),
                            "bones": bones[i].ravel().tolist(), "hips": hips[i].round(4).tolist(), "leg": round(float(legs[i]), 4)} for i in range(len(kp2d))]
                if self.vmc:
                    self.vmc.send(humanoid.local(world[:1])[0], hips[0], time.monotonic())
            t2 = time.perf_counter()
        return {"type": "frame", "w": W, "h": H, "persons": persons, "det_ms": (t1 - t0) * 1000, "hmr_ms": (t2 - t1) * 1000}, verts

    def faces(self):
        with torch.inference_mode():
            return self.model("hmr", self.settings["model"]).model.head_pose.faces_np().astype(np.int32).tobytes()

    def close(self):
        self.set_vmc("")


rooms = {}
# Shared by all rooms so two live views never hold two copies; dropped when the last room closes.
models = {}


def write_video(frames):
    """frames: [(t, jpeg)] at their arrival times; the clip plays at their average rate."""
    root = os.path.join(folder_paths.get_input_directory(), RECORD_DIR)
    os.makedirs(root, exist_ok=True)
    name = f"live_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    fps = (len(frames) - 1) / max(frames[-1][0] - frames[0][0], 1e-3) if len(frames) > 1 else 30.0
    first = Image.open(BytesIO(frames[0][1]))
    w, h = first.width // 2 * 2, first.height // 2 * 2
    with av.open(os.path.join(root, name), "w") as container:
        stream = container.add_stream("h264", rate=max(1, round(fps)))
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        for _, jpeg in frames:
            rgb = np.array(Image.open(BytesIO(jpeg)).convert("RGB"))[:h, :w]
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return f"{RECORD_DIR}/{name}", len(frames), fps


def send_latest(viewers, busy, data):
    """Frames to viewers never hold up the camera: a viewer still receiving the previous frame skips this one."""
    for ws in list(viewers):
        if ws in busy:
            continue
        busy.add(ws)

        async def send(ws=ws):
            try:
                await ws.send_bytes(data)
            except ConnectionResetError:
                viewers.discard(ws)
            finally:
                busy.discard(ws)
        asyncio.create_task(send())


async def send_all(sockets, data):
    for ws in list(sockets):
        try:
            await ws.send_bytes(data) if isinstance(data, bytes) else await ws.send_str(data)
        except ConnectionResetError:
            sockets.discard(ws)


def register_routes(routes):
    @routes.get("/instanthmr/camera")
    async def camera_page(request):
        return web.FileResponse(os.path.join(STATIC_DIR, "camera.html"))

    @routes.get("/instanthmr/static/{path:.+}")
    async def static_file(request):
        path = os.path.abspath(os.path.join(STATIC_DIR, request.match_info["path"]))
        if os.path.commonpath([path, STATIC_DIR]) != STATIC_DIR or not os.path.isfile(path):
            raise web.HTTPNotFound()
        return web.FileResponse(path)

    @routes.get("/instanthmr/lan")
    async def lan_addresses(request):
        addresses = {a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
        return web.json_response(sorted(a for a in addresses if not a.startswith("127.")))

    @routes.get("/instanthmr/recordings")
    async def recordings(request):
        return web.json_response(recording_files())

    @routes.get("/instanthmr/live")
    async def live(request):
        room_id, role = request.query.get("room", ""), request.query.get("role", "viewer")
        if not room_id.isalnum() or len(room_id) > 32:
            raise web.HTTPBadRequest()
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
        await ws.prepare(request)
        room = rooms.setdefault(room_id, Room())
        members = room.cameras if role == "camera" else room.viewers
        members.add(ws)
        await send_all(room.viewers, json.dumps({"type": "cameras", "count": len(room.cameras)}))
        loop = asyncio.get_running_loop()
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.BINARY and role == "camera":
                    error = None
                    if not room.settings["model"] or not room.settings["detector"]:
                        error = "Pick an InstantHMR model and an RT-DETR detector in the InstantHMR Live Capture node"
                    else:
                        jpeg = msg.data
                        received = time.monotonic()
                        try:
                            header, verts = await loop.run_in_executor(GPU, room.infer, jpeg)
                        except Exception as e:
                            error = str(e)
                    if error:
                        await ws.send_str(json.dumps({"type": "frame", "persons": [], "error": error}))
                        await send_all(room.viewers, json.dumps({"type": "error", "error": error}))
                        await asyncio.sleep(1)
                        continue
                    if room.recording is not None:
                        room.recording.append((received, jpeg))
                        header["recorded"] = len(room.recording)
                    await ws.send_str(json.dumps(header))
                    send_latest(room.viewers, room.sending, pack(header, jpeg, verts))
                elif msg.type == web.WSMsgType.TEXT and role == "viewer":
                    data = json.loads(msg.data)
                    if data["type"] == "settings":
                        vmc = room.settings["vmc"]
                        room.settings.update({k: data[k] for k in room.settings})
                        if room.settings["vmc"] != vmc:
                            try:
                                room.set_vmc(room.settings["vmc"])
                            except (ValueError, OSError) as e:
                                room.settings["vmc"] = ""
                                await ws.send_str(json.dumps({"type": "error", "error": f"VMC target: {e}"}))
                        if data["model"]:
                            bones = [[name, parent, child, aim] for name, parent, _, child, aim in BONES]
                            await ws.send_bytes(pack({"type": "faces", "bones": bones}, await loop.run_in_executor(GPU, room.faces)))
                    elif data["type"] == "record":
                        if data["on"]:
                            room.recording = []
                        elif room.recording:
                            frames, room.recording = room.recording, None
                            name, count, fps = await asyncio.to_thread(write_video, frames)
                            await send_all(room.viewers, json.dumps({"type": "recorded", "file": name, "frames": count, "fps": fps}))
                        else:
                            room.recording = None
        finally:
            members.discard(ws)
            if not room.viewers and not room.cameras:
                room.close()
                rooms.pop(room_id, None)
                if not rooms:
                    models.clear()
            else:
                await send_all(room.viewers, json.dumps({"type": "cameras", "count": len(room.cameras)}))
        return ws
