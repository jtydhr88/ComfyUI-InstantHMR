import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.ldm.sam3d_body.mhr.mhr_head import MHRHead
from comfy.ldm.sam3d_body.mhr.mhr_rig import MHRRig
from comfy.ldm.sam3d_body.mhr.mhr_utils import batchXYZfrom6D, compact_cont_to_model_params_body
from comfy.ops import cast_to_input

# HGNetV2 b0/b1/b2/b4 as in timm: stem (mid, out), stages (ic, mc, oc, blocks, downsample, light, k, layers), LAB
BACKBONES = {
    "nano": ((16, 16), [(16, 16, 64, 1, False, False, 3, 3), (64, 32, 256, 1, True, False, 3, 3), (256, 64, 512, 2, True, True, 5, 3), (512, 128, 1024, 1, True, True, 5, 3)], True),
    "tiny": ((24, 32), [(32, 32, 64, 1, False, False, 3, 3), (64, 48, 256, 1, True, False, 3, 3), (256, 96, 512, 2, True, True, 5, 3), (512, 192, 1024, 1, True, True, 5, 3)], True),
    "small": ((24, 32), [(32, 32, 96, 1, False, False, 3, 4), (96, 64, 384, 1, True, False, 3, 4), (384, 128, 768, 3, True, True, 5, 4), (768, 256, 1536, 1, True, True, 5, 4)], True),
    "medium": ((32, 48), [(48, 48, 128, 1, False, False, 3, 6), (128, 96, 512, 1, True, False, 3, 6), (512, 192, 1024, 3, True, True, 5, 6), (1024, 384, 2048, 1, True, True, 5, 6)], False),
}

NUM_KEYPOINTS = 70
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 4
KP2D_BINS = 96


def backbone_from_state_dict(sd):
    stem = sd["backbone.stem.stem1.conv.weight"].shape[0]
    stage1 = sd["backbone.stages.0.blocks.0.aggregation.1.conv.weight"].shape[0]
    for name, ((mid, _), stages, _) in BACKBONES.items():
        if mid == stem and stages[0][2] == stage1:
            return name
    raise ValueError(f"Unknown InstantHMR backbone (stem {stem}, stage1 {stage1})")


class ConvAct(nn.Module):
    # BatchNorm is folded into the conv in the released weights.
    def __init__(self, ic, oc, k, s=1, groups=1, use_act=True, use_lab=False, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv = operations.Conv2d(ic, oc, k, s, (k - 1) // 2, groups=groups, device=device, dtype=dtype)
        self.use_act = use_act
        self.lab = nn.Parameter(torch.empty(2, device=device, dtype=dtype)) if use_act and use_lab else None

    def forward(self, x):
        x = self.conv(x)
        if self.use_act:
            x = F.relu(x)
        if self.lab is not None:
            lab = cast_to_input(self.lab, x)
            x = x * lab[0] + lab[1]
        return x


class LightConvAct(nn.Module):
    def __init__(self, ic, oc, k, use_lab, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv1 = ConvAct(ic, oc, 1, use_act=False, use_lab=use_lab, device=device, dtype=dtype, operations=operations)
        self.conv2 = ConvAct(oc, oc, k, groups=oc, use_lab=use_lab, device=device, dtype=dtype, operations=operations)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Stem(nn.Module):
    def __init__(self, mc, oc, use_lab, device=None, dtype=None, operations=None):
        super().__init__()
        kw = dict(use_lab=use_lab, device=device, dtype=dtype, operations=operations)
        self.stem1 = ConvAct(3, mc, 3, 2, **kw)
        self.stem2a = ConvAct(mc, mc // 2, 2, **kw)
        self.stem2b = ConvAct(mc // 2, mc, 2, **kw)
        self.stem3 = ConvAct(mc * 2, mc, 3, 2, **kw)
        self.stem4 = ConvAct(mc, oc, 1, **kw)
        self.pool = nn.MaxPool2d(2, 1, ceil_mode=True)

    def forward(self, x):
        x = F.pad(self.stem1(x), (0, 1, 0, 1))
        x2 = self.stem2b(F.pad(self.stem2a(x), (0, 1, 0, 1)))
        return self.stem4(self.stem3(torch.cat([self.pool(x), x2], 1)))


class HGBlock(nn.Module):
    def __init__(self, ic, mc, oc, layer_num, k, residual, light, use_lab, device=None, dtype=None, operations=None):
        super().__init__()
        kw = dict(use_lab=use_lab, device=device, dtype=dtype, operations=operations)
        self.residual = residual
        if light:
            self.layers = nn.ModuleList([LightConvAct(ic if i == 0 else mc, mc, k, **kw) for i in range(layer_num)])
        else:
            self.layers = nn.ModuleList([ConvAct(ic if i == 0 else mc, mc, k, **kw) for i in range(layer_num)])
        self.aggregation = nn.Sequential(ConvAct(ic + layer_num * mc, oc // 2, 1, **kw), ConvAct(oc // 2, oc, 1, **kw))

    def forward(self, x):
        outs = [x]
        for layer in self.layers:
            outs.append(layer(outs[-1]))
        y = self.aggregation(torch.cat(outs, 1))
        return y + x if self.residual else y


class HGStage(nn.Module):
    def __init__(self, ic, mc, oc, blocks, downsample, light, k, layer_num, use_lab, device=None, dtype=None, operations=None):
        super().__init__()
        kw = dict(use_lab=use_lab, device=device, dtype=dtype, operations=operations)
        self.downsample = ConvAct(ic, ic, 3, 2, groups=ic, use_act=False, **kw) if downsample else nn.Identity()
        self.blocks = nn.Sequential(*[HGBlock(ic if i == 0 else oc, mc, oc, layer_num, k, i != 0, light, **kw) for i in range(blocks)])

    def forward(self, x):
        return self.blocks(self.downsample(x))


class HGNetV2(nn.Module):
    def __init__(self, stem, stages, use_lab, device=None, dtype=None, operations=None):
        super().__init__()
        self.stem = Stem(*stem, use_lab, device=device, dtype=dtype, operations=operations)
        self.stages = nn.Sequential(*[HGStage(*cfg, use_lab, device=device, dtype=dtype, operations=operations) for cfg in stages])
        self.out_channels = stages[-1][2]

    def forward(self, x):
        return self.stages(self.stem(x))


class Attention(nn.Module):
    def __init__(self, dim, heads, device=None, dtype=None, operations=None):
        super().__init__()
        self.heads = heads
        self.q = operations.Linear(dim, dim, device=device, dtype=dtype)
        self.kv = operations.Linear(dim, dim * 2, device=device, dtype=dtype)
        self.out = operations.Linear(dim, dim, device=device, dtype=dtype)

    def forward(self, x, context):
        k, v = self.kv(context).chunk(2, dim=-1)
        attention = optimized_attention_for_device(x.device, small_input=True)
        return self.out(attention(self.q(x), k, v, self.heads))


class DecoderLayer(nn.Module):
    # nn.TransformerDecoderLayer(norm_first=True, activation="gelu")
    def __init__(self, dim, heads, device=None, dtype=None, operations=None):
        super().__init__()
        kw = dict(device=device, dtype=dtype, operations=operations)
        self.norm1 = operations.LayerNorm(dim, device=device, dtype=dtype)
        self.self_attn = Attention(dim, heads, **kw)
        self.norm2 = operations.LayerNorm(dim, device=device, dtype=dtype)
        self.cross_attn = Attention(dim, heads, **kw)
        self.norm3 = operations.LayerNorm(dim, device=device, dtype=dtype)
        self.mlp = nn.Sequential(operations.Linear(dim, dim * 4, device=device, dtype=dtype), nn.GELU(),
                                 operations.Linear(dim * 4, dim, device=device, dtype=dtype))

    def forward(self, x, memory):
        h = self.norm1(x)
        x = x + self.self_attn(h, h)
        x = x + self.cross_attn(self.norm2(x), memory)
        return x + self.mlp(self.norm3(x))


class MHRBodyRig(MHRRig):
    # InstantHMR is trained on the MHR mesh without pose correctives, so their 663 MB weights are not loaded.
    def __init__(self, device=None):
        super().__init__(device=device)
        self.pose_corr_weight = None
        self.pose_corr_sparse_weight = None
        self.pose_corr_sparse_indices = None

    def forward(self, identity_coeffs, model_parameters, expr_coeffs, apply_correctives=False):
        return super().forward(identity_coeffs, model_parameters, expr_coeffs, apply_correctives=False)


class InstantHMRHead(MHRHead):
    """SAM 3D Body's continuous MHR head without the face block, with bounded body size."""

    def __init__(self, dim, mhr_rig, device=None, dtype=None, operations=None):
        super().__init__(dim, mhr_rig, device=device, dtype=dtype, operations=operations)
        self.npose -= self.num_face_comps
        self.proj = nn.Sequential(operations.LayerNorm(dim, device=device, dtype=dtype),
                                  operations.Linear(dim, dim, device=device, dtype=dtype), nn.GELU(),
                                  operations.Linear(dim, self.npose, device=device, dtype=dtype))
        self.init_estimate = nn.Parameter(torch.empty(1, self.npose, device=device), requires_grad=False)
        # (lo, half range) of the tanh bounds on the 6 *_flexible body channels and the 28 bone-scale coefficients
        self.flex_bounds = nn.Parameter(torch.empty(2, 6, device=device), requires_grad=False)
        self.scale_bounds = nn.Parameter(torch.empty(2, self.num_scale_comps, device=device), requires_grad=False)

    def params(self, x):
        """(global_rot, body_pose, shape, scale, hand) that decode() turns into the posed MHR body."""
        pred = self.proj(x).float() + self.init_estimate
        B = pred.shape[0]
        root, body, shape, scale, hand = pred.split([6, self.body_cont_dim, self.num_shape_comps, self.num_scale_comps, self.num_hand_comps * 2], dim=1)

        global_rot = batchXYZfrom6D(root)
        body_pose = compact_cont_to_model_params_body(body)
        body_pose[:, 124:130] = self.flex_bounds[0] + self.flex_bounds[1] * (body_pose[:, 124:130].tanh() + 1)
        scale = self.scale_bounds[0] + self.scale_bounds[1] * (scale.tanh() + 1)
        return global_rot, body_pose, shape, scale, hand

    def decode(self, global_rot, body_pose, shape, scale, hand):
        face = global_rot.new_zeros(global_rot.shape[0], self.num_face_comps)

        verts, j3d, jcoords, model_params, joint_rots = self.mhr_forward(
            global_trans=torch.zeros_like(global_rot), global_rot=global_rot, body_pose_params=body_pose,
            hand_pose_params=hand, scale_params=scale, shape_params=shape, expr_params=face,
            return_keypoints=True, return_joint_coords=True, return_model_params=True, return_joint_rotations=True,
        )
        j3d = j3d[:, :NUM_KEYPOINTS]
        for t in (verts, j3d, jcoords):
            t[..., [1, 2]] *= -1
        return {
            "global_rot": global_rot,
            "body_pose": body_pose,
            "shape": shape,
            "scale": scale,
            "hand": hand,
            "face": face,
            "pred_keypoints_3d": j3d,
            "pred_vertices": verts,
            "pred_joint_coords": jcoords,
            "joint_global_rots": joint_rots,
            "mhr_model_params": model_params,
        }


class InstantHMR(nn.Module):
    image_size = 288

    def __init__(self, backbone, device=None, dtype=None, operations=None):
        super().__init__()
        stem, stages, use_lab = BACKBONES[backbone]
        self.backbone = HGNetV2(stem, stages, use_lab, device=device, dtype=dtype, operations=operations)
        self.feat_proj = operations.Linear(self.backbone.out_channels, D_MODEL, device=device, dtype=dtype)
        tokens = (self.image_size // 32) ** 2
        self.mem_pos_embed = nn.Parameter(torch.empty(1, tokens, D_MODEL, device=device, dtype=dtype), requires_grad=False)
        self.query_embed = nn.Parameter(torch.empty(1, 1 + NUM_KEYPOINTS, D_MODEL, device=device, dtype=dtype), requires_grad=False)
        self.cond_proj = nn.Sequential(operations.Linear(3, 128, device=device, dtype=dtype), nn.GELU(),
                                       operations.Linear(128, D_MODEL, device=device, dtype=dtype))
        self.layers = nn.ModuleList([DecoderLayer(D_MODEL, N_HEADS, device=device, dtype=dtype, operations=operations) for _ in range(N_LAYERS)])
        self.mhr = MHRBodyRig(device=device)
        self.head_pose = InstantHMRHead(D_MODEL, self.mhr, device=device, dtype=dtype, operations=operations)
        self.head_cam = operations.Linear(D_MODEL, 3, device=device, dtype=dtype)
        self.head_2d = nn.Sequential(operations.Linear(D_MODEL, 256, device=device, dtype=dtype), nn.GELU(),
                                     operations.Linear(256, 2 * KP2D_BINS, device=device, dtype=dtype))
        self.kp2d_bin_centers = nn.Parameter(torch.empty(KP2D_BINS, device=device), requires_grad=False)

    def forward(self, image, cliff_cond):
        """image (B, 3, 288, 288) ImageNet-normalised crop, cliff_cond (B, 3) box position in the frame."""
        params, cam_t, kp2d = self.predict(image, cliff_cond)
        return {**self.head_pose.decode(*params), "pred_cam_t": cam_t, "pred_keypoints_2d_crop": kp2d}

    def predict(self, image, cliff_cond):
        """The pose parameters, camera translation and crop 2D keypoints, before the MHR body is decoded."""
        memory = self.feat_proj(self.backbone(image).flatten(2).transpose(1, 2))
        memory = memory + cast_to_input(self.mem_pos_embed, memory)
        x = cast_to_input(self.query_embed, memory) + self.cond_proj(cliff_cond).unsqueeze(1)
        for layer in self.layers:
            x = layer(x, memory)

        logits = self.head_2d(x[:, 1:]).float().unflatten(-1, (2, KP2D_BINS))
        return self.head_pose.params(x[:, 0]), self.head_cam(x[:, 0]).float(), (logits.softmax(-1) * self.kp2d_bin_centers).sum(-1)

    def memory_used_forward(self, batch):
        return batch * 64 * 1024 * 1024 * comfy.model_management.dtype_size(self.feat_proj.weight.dtype)
