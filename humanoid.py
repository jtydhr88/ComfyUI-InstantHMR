import socket
import struct

import numpy as np
import torch

UP, DOWN, LEFT, RIGHT = (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)


def _side(s, out, j):
    """j: clavicle, upper arm, forearm, hand, thigh, shin, foot, toes, then the first joint of thumb, index, middle, ring, little."""
    clav, upper, fore, hand, thigh, shin, foot, toes, *fingers = j
    S = "Left" if out == LEFT else "Right"
    bones = [
        (f"{S}Shoulder", "UpperChest", clav, f"{S}UpperArm", out),
        (f"{S}UpperArm", f"{S}Shoulder", upper, f"{S}LowerArm", out),
        (f"{S}LowerArm", f"{S}UpperArm", fore, f"{S}Hand", out),
        (f"{S}Hand", f"{S}LowerArm", hand, f"{S}MiddleProximal", out),
        (f"{S}UpperLeg", "Hips", thigh, f"{S}LowerLeg", DOWN),
        (f"{S}LowerLeg", f"{S}UpperLeg", shin, f"{S}Foot", DOWN),
        (f"{S}Foot", f"{S}LowerLeg", foot, None, None),
        (f"{S}Toes", f"{S}Foot", toes, None, None),
    ]
    for finger, first in zip(("Thumb", "Index", "Middle", "Ring", "Little"), fingers):
        # Thumbs keep their rest angle to the hand; the other fingers are straight along the arm in a T-pose.
        aim = None if finger == "Thumb" else out
        parent = f"{S}Hand"
        for k, seg in enumerate(("Proximal", "Intermediate", "Distal")):
            child = f"{S}{finger}{('Intermediate', 'Distal')[k]}" if k < 2 else None
            bones.append((f"{S}{finger}{seg}", parent, first + k, child, aim if child else None))
            parent = f"{S}{finger}{seg}"
    return bones


# Unity humanoid bones: (name, parent, MHR joint, child bone the T-pose direction is measured to, T-pose direction).
# A bone without a direction keeps its rest angle to its parent.
BONES = [
    ("Hips", None, 1, None, None),
    ("Spine", "Hips", 34, "Chest", UP),
    ("Chest", "Spine", 36, "UpperChest", UP),
    ("UpperChest", "Chest", 37, "Neck", UP),
    ("Neck", "UpperChest", 110, "Head", UP),
    ("Head", "Neck", 113, None, None),
    *_side("Left", LEFT, (74, 75, 76, 78, 2, 3, 6, 7, 96, 92, 88, 84, 80)),
    *_side("Right", RIGHT, (38, 39, 40, 42, 18, 19, 22, 23, 60, 56, 52, 48, 44)),
]
NAMES = [b[0] for b in BONES]
INDEX = {n: i for i, n in enumerate(NAMES)}
JOINTS = [b[2] for b in BONES]
PARENTS = [INDEX[b[1]] if b[1] else -1 for b in BONES]


def swing(a, b):
    """Rotation matrix taking unit vector a onto unit vector b along the shortest arc."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def mat_to_quat(m):
    """(..., 3, 3) rotation matrices to (..., 4) xyzw quaternions."""
    m = np.asarray(m, dtype=np.float64)
    w = np.sqrt(np.maximum(0, 1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2])) / 2
    x = np.sqrt(np.maximum(0, 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2])) / 2
    y = np.sqrt(np.maximum(0, 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2])) / 2
    z = np.sqrt(np.maximum(0, 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2])) / 2
    x = np.copysign(x, m[..., 2, 1] - m[..., 1, 2])
    y = np.copysign(y, m[..., 0, 2] - m[..., 2, 0])
    z = np.copysign(z, m[..., 1, 0] - m[..., 0, 1])
    return np.stack([x, y, z, w], -1)


class HumanoidRig:
    """Turns MHR joint rotations into humanoid bone rotations relative to a T-pose whose bones all have identity
    orientation (the convention of VRM normalized bones and VMC), in the MHR frame: Y up, facing +Z, left at +X."""

    def __init__(self, mhr):
        with torch.inference_mode():
            p = next(mhr.parameters())
            _, state = mhr(p.new_zeros(1, mhr.NUM_IDENTITY), p.new_zeros(1, 204), p.new_zeros(1, mhr.NUM_EXPR))
        state = state[0].double().cpu().numpy()
        pos = state[:, :3]
        x, y, z, w = state[:, 3:7].T
        rest = np.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)[JOINTS]
        # T-pose world orientation of each bone: its rest orientation swung onto the T-pose direction.
        to_t = []
        for name, parent, joint, child, aim in BONES:
            if aim is None:
                to_t.append(to_t[INDEX[parent]] if parent else np.eye(3))
            else:
                d = pos[BONES[INDEX[child]][2]] - pos[joint]
                to_t.append(swing(d / np.linalg.norm(d), np.array(aim, dtype=np.float64)))
        # W = R · R_rest^T · S^T, with the constant R_rest^T · S^T folded per bone.
        self.offset = np.transpose(np.stack(to_t) @ rest, (0, 2, 1))

    def world(self, joint_rots):
        """joint_rots (B, 127, 3, 3) MHR global joint rotations -> (B, bones, 3, 3)."""
        r = joint_rots[:, JOINTS].double().cpu().numpy()
        return r @ self.offset

    @staticmethod
    def local(world):
        parent = world[:, [max(p, 0) for p in PARENTS]]
        local = np.transpose(parent, (0, 1, 3, 2)) @ world
        local[:, 0] = world[:, 0]
        return local


def _osc_str(s):
    b = s.encode() + b"\0"
    return b + b"\0" * (-len(b) % 4)


def _osc(address, *args):
    tags = "," + "".join("s" if isinstance(a, str) else "i" if isinstance(a, int) else "f" for a in args)
    data = b"".join(_osc_str(a) if isinstance(a, str) else struct.pack(">i" if isinstance(a, int) else ">f", a) for a in args)
    return _osc_str(address) + _osc_str(tags) + data


class VMCSender:
    """Sends one performer to a VMC protocol receiver (VSeeFace, Warudo, VNyan, VMC4B...) over UDP."""

    def __init__(self, target):
        host, _, port = target.rpartition(":")
        self.address = (host or "127.0.0.1", int(port))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.start = None

    def send(self, local, hips, t):
        """local (bones, 3, 3) humanoid local rotations, hips (3,) hips position in metres, both in the MHR frame."""
        # The MHR frame becomes Unity's left-handed frame by mirroring X.
        q = mat_to_quat(local)
        q[:, 1:3] *= -1
        if self.start is None:
            self.start = hips.copy()
        move = hips - self.start
        messages = [_osc("/VMC/Ext/OK", 1), _osc("/VMC/Ext/T", float(t)), _osc("/VMC/Ext/Root/Pos", "root", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)]
        for i, name in enumerate(NAMES):
            p = (-float(move[0]), 0.9 + float(move[1]), 0.0) if i == 0 else (0.0, 0.0, 0.0)
            messages.append(_osc("/VMC/Ext/Bone/Pos", name, *p, *map(float, q[i])))
        packet = _osc_str("#bundle") + struct.pack(">Q", 1) + b"".join(struct.pack(">i", len(m)) + m for m in messages)
        self.socket.sendto(packet, self.address)

    def close(self):
        self.socket.close()
