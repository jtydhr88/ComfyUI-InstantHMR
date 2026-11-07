English | [中文](README.zh-CN.md)

# ComfyUI-InstantHMR

[InstantHMR](https://github.com/Naturalpad-studio/InstantHMR) recovers the 3D body of every person in an image or video: pose, body shape, 70 keypoints in 2D and 3D, and the full [MHR](https://github.com/facebookresearch/MHR) body mesh. It is a small, fast model (20-32 M parameters) that outputs the same MHR body as ComfyUI's SAM 3D Body, so its result plugs into the native SAM 3D Body nodes for smoothing, rendering and 3D export.

![InstantHMR on four dancers](docs/render_friends.jpg)

## Nodes

| Node | What it does |
| --- | --- |
| `Load InstantHMR Model` | loads an InstantHMR checkpoint from `models/detection` |
| `Run InstantHMR Prediction` | predicts one MHR body per person box and outputs `MHR_POSE_DATA` |
| `InstantHMR Live Capture` | live 2D and 3D preview from a camera that can drive a rigged character or a VMC app, with recording; outputs the recorded clip as `VIDEO` |

`Run InstantHMR Prediction` takes person boxes from `RT-DETR Detect` (class `person`) or tracks from `SAM3 Video Track`; with neither, the whole frame is one person. `fov` is the vertical field of view in degrees; connect `MoGe` through `MoGe Geometry To FOV` for the camera of a video, or leave it at 0 for a focal of 1.05x the image diagonal.

The output works with the native nodes:

- `Smooth SAM3D Body Pose Data` for video jitter
- `Render 3D Body Pose`: mesh, silhouette, OpenPose 2D/3D, SCAIL
- `Create 3D Animation File`: animated GLB in the `openpose` and `scail` styles. The `body_mesh` and `bones_only` styles and BVH read the MHR rig from a `sam3d_body_model` input, so connect a SAM 3D Body loader for those.

![Example workflow](docs/workflow_image.jpg)

## Live capture

![Live capture](docs/live_capture.jpg)

`InstantHMR Live Capture` runs RT-DETR and InstantHMR on a camera feed while you watch: the camera image with the 2D skeletons on the left, the 3D bodies on the right (drag to orbit). Pick the InstantHMR `model`, the RT-DETR `detector` and the `camera` in the node, then press `Start`. These settings are saved with the workflow.

- `camera` **This device** uses the webcam of the browser showing ComfyUI.
- `camera` **Phone / other device** shows a QR code. Scan it with a phone on the same network; the phone page streams its camera and shows the skeletons, while the node shows the 3D view.
- `● Record` records the incoming frames to `input/instanthmr_live/`. The new clip is selected in the `recording` list, and running the workflow outputs it as `VIDEO`, ready for `Get Video Components` → `Run InstantHMR Prediction` and the rest of the pipeline (`example_workflows/instanthmr_live.json`).
- `detect_every` (advanced input) runs the detector every N frames and reuses its boxes in between; the detector is the slowest step, so 2-3 keeps the preview smooth. `fov` (advanced input) works like the prediction node's `fov`.
- `character` lists rigged characters (`.fbx`, `.glb`, `.gltf`, `.vrm`) found in the `input` folder, for example `input/3d/`. Pick one and the 3D view drives it instead of the MHR body, one copy per person. Bones are matched by name for Mixamo, Unreal and similar rigs, and from the humanoid table for VRM. T-pose and A-pose characters both work; the character is scaled to the performer's leg length.
- `vmc`: type `host:port` of an app that receives the [VMC protocol](https://protocol.vmc.info/) (VSeeFace, Warudo, VNyan, [VMC Link](https://extensions.blender.org/add-ons/vmc-link/) for Blender, VMC4U for Unity) to stream the first person as a humanoid avatar pose, for example `127.0.0.1:39539`. ComfyUI sends it, so the address is seen from the computer running ComfyUI. Clear the field to stop.
  - In Blender: install VMC Link, import a T-pose character (for example a Mixamo FBX), apply its rotation and scale (Object > Apply > All Transforms), pick the armature in the VMC Link panel and press Start Receiver. Start Recording there keys the motion onto the armature. For Mixamo rigs, set these in VMC Link's manual bone mapping (Advanced): UpperLeg → UpLeg, LowerLeg → Leg, Toes → ToeBase, Chest → Spine1, UpperChest → Spine2, and the fingers → HandThumb1–3, HandIndex1–3 and so on (Little → HandPinky).

Browsers open a camera only on `localhost` or over HTTPS. For a phone, start ComfyUI with a certificate, for example a self-signed one:

```
openssl req -x509 -newkey rsa:2048 -nodes -days 825 -keyout key.pem -out cert.pem -subj "/CN=ComfyUI" -addext "subjectAltName=IP:192.168.0.10"
python main.py --listen 0.0.0.0 --tls-keyfile key.pem --tls-certfile cert.pem
```

Put your computer's LAN address in `subjectAltName`, open ComfyUI at `https://<address>:8188`, and accept the certificate warning once on the computer and once on the phone.

## Models

| File | Backbone | 3DPW PA-MPJPE (mm) | Licence |
| --- | --- | --- | --- |
| `instanthmr_nano.safetensors` | HGNetV2-B0 | 46.6 | CC BY-NC 4.0 |
| `instanthmr_tiny.safetensors` | HGNetV2-B1 | 44.0 | CC BY-NC 4.0 |
| `instanthmr_small.safetensors` | HGNetV2-B2 | 41.7 | CC BY-NC 4.0 |
| `instanthmr_medium.safetensors` | HGNetV2-B4 | 40.0 | CC BY-NC 4.0 |
| `instanthmr_medium_commercial.safetensors` | HGNetV2-B4 | 50.7 | Apache-2.0 |

Download them from [Hugging Face](https://huggingface.co/jtydhr88/ComfyUI-InstantHMR) and put them in `models/detection`. Each file holds the network and the MHR body rig it is trained on, so no other model is needed. The CC BY-NC weights are for non-commercial use; `medium_commercial` is trained on commercially usable images only.

### Converting the released ONNX files

The upstream release ships ONNX graphs on [Hugging Face](https://huggingface.co/momolesang/InstantHMR). `scripts/convert_onnx.py` turns one into a checkpoint for this pack, taking the MHR rig from ComfyUI's SAM 3D Body checkpoint ([`sam_3d_body_dinov3_bf16.safetensors`](https://huggingface.co/Comfy-Org/sam-3d-body)). It needs `onnx`, and checks the result against the ONNX graph when `onnxruntime` is installed. From the ComfyUI folder:

```
pip install onnx onnxruntime
python custom_nodes/ComfyUI-InstantHMR/scripts/convert_onnx.py instanthmr-medium.onnx \
    models/detection/sam_3d_body_dinov3_bf16.safetensors models/detection/instanthmr_medium.safetensors \
    --body mhr-body.onnx
```

## Example workflows

- `example_workflows/instanthmr_image.json`: RT-DETR boxes on a photo, rendered as meshes over the image.
- `example_workflows/instanthmr_video.json`: the SAM 3D Body video template with InstantHMR: SAM3 tracks, MoGe FoV, smoothing, a rendered video and a SCAIL GLB.
- `example_workflows/instanthmr_live.json`: the same pipeline on a clip recorded with `InstantHMR Live Capture`.

## Tips

- InstantHMR is trained on whole people. On close-ups that cut the body at the chest the pose stays good, while the mesh can sit a few percent off the person in the image; full-body shots line up closely.
- For several people in a video, use SAM3 tracks so each person keeps the same index across frames, which the smoothing node relies on.
- Smaller models (`nano`, `tiny`) are faster and less accurate; the network takes about 1 ms per person on a desktop GPU, so the detector and tracker set the speed of a workflow.

## Credits

InstantHMR by Mohamed Adjel, Kahina Chalabi and Vincent Bonnet ([paper](https://hal.science/hal-05854387)), Apache-2.0 code. The body model is Meta's [MHR](https://github.com/facebookresearch/MHR) (Apache-2.0). Demo footage from [Mixkit](https://mixkit.co/).
