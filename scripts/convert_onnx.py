"""Convert a released InstantHMR ONNX graph into a ComfyUI .safetensors checkpoint.

The checkpoint holds the network, with the BatchNorms folded into the convs as in the ONNX graph,
plus the MHR rig (without pose correctives) and the MHR head tables, read from the SAM 3D Body
checkpoint ComfyUI already uses. Run from the ComfyUI folder:

    python custom_nodes/ComfyUI-InstantHMR/scripts/convert_onnx.py instanthmr-medium.onnx \
        models/detection/sam_3d_body_dinov3_bf16.safetensors models/detection/instanthmr_medium.safetensors

With onnxruntime installed, the converted model is checked against the ONNX graph on random crops,
and against mhr-body.onnx when --body is given.
"""
import argparse
import os
import sys

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comfy.ops  # noqa: E402
from instanthmr_model import BACKBONES, ConvAct, InstantHMR  # noqa: E402

RIG_SKIP = {"mhr.pose_corr_weight", "mhr.pose_corr_sparse_weight", "mhr.pose_corr_sparse_indices"}
HEAD_KEYS = ["faces", "keypoint_mapping", "scale_mean", "scale_comps", "hand_pose_mean", "hand_pose_comps",
             "hand_joint_idxs_left", "hand_joint_idxs_right", "joint_rotation", "right_wrist_coords", "root_coords",
             "local_to_world_wrist", "nonhand_param_idxs", "face_region_rgb"]


def onnx_tensors(graph):
    values = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    for n in graph.node:
        if n.op_type == "Constant":
            values[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    consumers = {}
    for n in graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    return values, consumers


def next_op(consumers, node, op_type):
    for n in consumers.get(node.output[0], []):
        if n.op_type == op_type:
            return n
    return None


def scalar_input(values, node):
    return next(float(values[i]) for i in node.input if i in values and values[i].size == 1)


def convert_network(graph):
    values, consumers = onnx_tensors(graph)

    convs = []
    for n in graph.node:
        if n.op_type != "Conv":
            continue
        relu = next_op(consumers, n, "Relu")
        lab = None
        if relu is not None:
            mul = next_op(consumers, relu, "Mul")
            if mul is not None:
                lab = (scalar_input(values, mul), scalar_input(values, next_op(consumers, mul, "Add")))
        convs.append((values[n.input[1]], values[n.input[2]], relu is not None, lab))

    sd = {k[4:]: v for k, v in values.items() if k.startswith("net.")}
    for n in graph.node:
        if n.op_type == "MatMul" and n.input[1] in values and not n.input[1].startswith("net."):
            add = next_op(consumers, n, "Add")
            bias = next((i for i in add.input if i.startswith("net.")), None) if add is not None else None
            if bias is not None:
                sd[bias[4:].replace("bias", "weight")] = values[n.input[1]].T
        if n.op_type == "Mul" and n.input[0] in values and n.input[0].startswith("onnx::"):
            add = next_op(consumers, n, "Add")
            lo = next((i for i in add.input if i.startswith("net.cont_head.")), None) if add is not None else None
            if lo is not None:
                sd[lo[4:].replace("_lo", "_half_range")] = values[n.input[0]]
        if n.op_type == "Softmax" and n.input[0] == "/net/Cast_output_0":
            mul = next_op(consumers, n, "Mul")
            sd["kp2d_bin_centers"] = values[next(i for i in mul.input if i in values)]
    return convs, sd


def build_state_dict(onnx_path, sam3d_path):
    convs, src = convert_network(onnx.load(onnx_path).graph)
    sd = {}

    stem_mid = convs[0][0].shape[0]
    stage1 = None
    backbone = None
    for name, ((mid, _), stages, _) in BACKBONES.items():
        if mid == stem_mid and len(convs) == count_convs(stages):
            stage1 = stages[0][2]
            backbone = name
    if backbone is None:
        raise ValueError(f"no HGNetV2 config matches stem {stem_mid} with {len(convs)} convs")

    model = InstantHMR(backbone, device="meta", operations=comfy.ops.disable_weight_init)
    modules = [(name, m) for name, m in model.backbone.named_modules() if isinstance(m, ConvAct)]
    assert len(modules) == len(convs)
    for (name, m), (w, b, act, lab) in zip(modules, convs):
        assert tuple(m.conv.weight.shape) == w.shape and m.use_act == act and (m.lab is not None) == (lab is not None), name
        sd[f"backbone.{name}.conv.weight"] = w
        sd[f"backbone.{name}.conv.bias"] = b
        if lab is not None:
            sd[f"backbone.{name}.lab"] = np.array(lab, dtype=np.float32)
    assert sd["backbone.stages.0.blocks.0.aggregation.1.conv.weight"].shape[0] == stage1

    for k in ("query_embed", "mem_pos_embed", "kp2d_bin_centers", "head_cam.weight", "head_cam.bias"):
        sd[k] = src[k]
    for p in ("weight", "bias"):
        sd[f"feat_proj.{p}"] = src[f"feat_proj.{p}"]
        sd[f"cond_proj.0.{p}"] = src[f"cond_proj.0.{p}"]
        sd[f"cond_proj.2.{p}"] = src[f"cond_proj.2.{p}"]
        sd[f"head_2d.0.{p}"] = src[f"head_2d_feat.0.{p}"]
        sd[f"head_2d.2.{p}"] = src[f"head_2d_logits.{p}"]
        sd[f"head_pose.proj.0.{p}"] = src[f"cont_head.norm.{p}"]
        sd[f"head_pose.proj.1.{p}"] = src[f"cont_head.mlp.0.{p}"]
        sd[f"head_pose.proj.3.{p}"] = src[f"cont_head.mlp.2.{p}"]
        for i in range(len(model.layers)):
            t, o = f"transformer.layers.{i}.", f"layers.{i}."
            for norm in ("norm1", "norm2", "norm3"):
                sd[o + f"{norm}.{p}"] = src[t + f"{norm}.{p}"]
            sd[o + f"self_attn.out.{p}"] = src[t + f"self_attn.out_proj.{p}"]
            sd[o + f"cross_attn.out.{p}"] = src[t + f"multihead_attn.out_proj.{p}"]
            sd[o + f"mlp.0.{p}"] = src[t + f"linear1.{p}"]
            sd[o + f"mlp.2.{p}"] = src[t + f"linear2.{p}"]
            for attn, name in (("self_attn", "self_attn"), ("multihead_attn", "cross_attn")):
                w = src[t + f"{attn}.in_proj_{p}"]
                d = w.shape[0] // 3
                sd[o + f"{name}.q.{p}"] = w[:d]
                sd[o + f"{name}.kv.{p}"] = w[d:]
    sd["head_pose.init_estimate"] = src["cont_head.init_estimate"]
    sd["head_pose.flex_bounds"] = np.stack([src["cont_head.flex_lo"], src["cont_head.flex_half_range"]])
    sd["head_pose.scale_bounds"] = np.stack([src["cont_head.coeff_lo"], src["cont_head.coeff_half_range"]])

    sd = {k: torch.from_numpy(np.ascontiguousarray(v).copy()) for k, v in sd.items()}
    with safe_open(sam3d_path, "pt") as f:
        for k in f.keys():
            if k.startswith("mhr.") and k not in RIG_SKIP:
                sd[k] = f.get_tensor(k)
        for k in HEAD_KEYS:
            sd[f"head_pose.{k}"] = f.get_tensor(f"head_pose.{k}")

    for ours, theirs in (("scale_mean", "scale_mean"), ("scale_comps", "scale_comps"), ("hand_pose_mean", "hand_mean"), ("hand_pose_comps", "hand_comps")):
        diff = (sd[f"head_pose.{ours}"] - torch.from_numpy(src[f"cont_head.{theirs}"])).abs().max().item()
        assert diff < 1e-6, f"MHR head table {ours} differs from the ONNX graph by {diff}"
    return backbone, sd


def count_convs(stages):
    n = 5
    for _, _, _, blocks, downsample, light, _, layers in stages:
        n += int(downsample) + blocks * (layers * (2 if light else 1) + 2)
    return n


def check(onnx_path, body_path, sd, backbone):
    import onnxruntime as ort
    model = InstantHMR(backbone, operations=comfy.ops.disable_weight_init)
    model.load_state_dict(sd, strict=True)
    model.eval()

    rng = np.random.default_rng(0)
    image = rng.standard_normal((4, 3, 288, 288)).astype(np.float32)
    cliff = (rng.standard_normal((4, 3)) * 0.3).astype(np.float32)
    names = ["mhr_params", "shape_params", "cam_trans", "joints_2d"]
    ref = dict(zip(names, ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(names, {"image": image, "cliff_cond": cliff})))
    with torch.inference_mode():
        out = model(torch.from_numpy(image), torch.from_numpy(cliff))
    ours = {"mhr_params": out["mhr_model_params"], "shape_params": out["shape"], "cam_trans": out["pred_cam_t"], "joints_2d": out["pred_keypoints_2d_crop"]}
    for k in names:
        print(f"{k:13s} max abs diff {np.abs(ours[k].numpy() - ref[k]).max():.2e}")
    if body_path:
        body = ort.InferenceSession(body_path, providers=["CPUExecutionProvider"])
        j3d, verts = body.run(["joints_3d", "vertices"], {"mhr_params": ref["mhr_params"], "shape_params": ref["shape_params"]})
        print(f"{'joints_3d':13s} max abs diff {np.abs(out['pred_keypoints_3d'].numpy() - j3d).max():.2e} m")
        print(f"{'vertices':13s} max abs diff {np.abs(out['pred_vertices'].numpy() - verts).max():.2e} m")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx")
    ap.add_argument("sam3d_body", help="ComfyUI SAM 3D Body checkpoint holding the MHR rig")
    ap.add_argument("output")
    ap.add_argument("--body", help="mhr-body.onnx, to check the mesh and keypoints too")
    args = ap.parse_args()

    backbone, sd = build_state_dict(args.onnx, args.sam3d_body)
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("onnxruntime not installed, skipping the check")
    else:
        check(args.onnx, args.body, sd, backbone)
    save_file(sd, args.output)
    print(f"wrote {args.output} ({backbone}, {os.path.getsize(args.output) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
