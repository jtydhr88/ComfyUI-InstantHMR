import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { CameraStreamer, drawPersons, liveUrl } from "/instanthmr/static/camera_client.js";

const STYLE = `
.ihmr-live { display: flex; flex-direction: column; gap: 6px; height: 100%; min-height: 0; font-size: 12px; color: var(--input-text, #ddd); }
.ihmr-bar { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.ihmr-bar select, .ihmr-bar input, .ihmr-bar button { background: var(--comfy-input-bg, #222); color: inherit; border: 1px solid var(--border-color, #444); border-radius: 6px; padding: 3px 6px; font: inherit; }
.ihmr-bar input { width: 52px; }
.ihmr-bar input.vmc { width: 120px; }
.ihmr-bar button { cursor: pointer; }
.ihmr-bar button.rec { color: #f87171; }
.ihmr-bar button.rec.on { background: #b91c1c; color: #fff; }
.ihmr-views { flex: 1; min-height: 0; display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.ihmr-view { position: relative; min-height: 0; background: #0b0b0b; border-radius: 6px; overflow: hidden; }
.ihmr-view canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
.ihmr-remote { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; text-align: center; padding: 8px; pointer-events: none; }
.ihmr-remote img { width: 140px; image-rendering: pixelated; background: #fff; padding: 6px; border-radius: 6px; pointer-events: auto; }
.ihmr-remote a { color: #93c5fd; word-break: break-all; pointer-events: auto; }
.ihmr-status { opacity: 0.8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
`;

function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  Object.assign(e, attrs);
  e.append(...children);
  return e;
}

async function listModels(folder, test) {
  const files = await (await api.fetchApi(`/models/${folder}`)).json();
  return files.filter((f) => test.test(f));
}

async function phoneUrl(room) {
  let host = location.host;
  if (["localhost", "127.0.0.1", "[::1]"].includes(location.hostname)) {
    const ips = await (await api.fetchApi("/instanthmr/lan")).json();
    if (ips.length) host = `${ips[0]}${location.port ? `:${location.port}` : ""}`;
  }
  return `https://${host}/instanthmr/camera?room=${room}`;
}

class LiveView {
  constructor(node) {
    this.node = node;
    this.settings = node.properties.instanthmr_live ??= {
      room: Math.random().toString(36).slice(2, 10),
      source: "remote", model: "", detector: "", fov: 0, detect_every: 2,
    };
    this.settings.character ??= "";
    this.settings.vmc ??= "";
    this.faces = null;
    this.meshes = [];
    this.characters = [];
    this.build();
  }

  build() {
    const s = this.settings;
    this.model = el("select", { title: "InstantHMR model" });
    this.detector = el("select", { title: "Person detector" });
    this.source = el("select", { title: "Camera" }, [
      el("option", { value: "remote", textContent: "Phone / other device" }),
      el("option", { value: "local", textContent: "This device" }),
    ]);
    this.source.value = s.source;
    this.fov = el("input", { type: "number", min: 0, max: 179, step: 1, value: s.fov, title: "Vertical FoV in degrees, 0 = 1.05x diagonal focal" });
    this.every = el("input", { type: "number", min: 1, max: 10, step: 1, value: s.detect_every, title: "Run the detector every N frames" });
    this.character = el("select", { title: "Body shown in the 3D view: the MHR mesh, or a rigged character from the input folder (FBX, glTF, VRM)" });
    this.vmc = el("input", { className: "vmc", value: s.vmc, placeholder: "VMC host:port", title: "Send the first person to a VMC protocol app (VSeeFace, Warudo, VNyan...), e.g. 127.0.0.1:39539. Empty = off" });
    this.startBtn = el("button", { textContent: "Start" });
    this.recBtn = el("button", { className: "rec", textContent: "● Rec", disabled: true });
    this.status = el("div", { className: "ihmr-status", textContent: "Stopped" });
    this.video = el("video", { muted: true, playsInline: true });
    this.view2d = el("canvas");
    this.canvas3d = el("canvas");
    this.remote = el("div", { className: "ihmr-remote" });
    this.root = el("div", { className: "ihmr-live" }, [
      el("div", { className: "ihmr-bar" }, [this.model, this.detector, this.source, "FoV", this.fov, "Detect every", this.every, this.character, this.vmc, this.startBtn, this.recBtn]),
      el("div", { className: "ihmr-views" }, [
        el("div", { className: "ihmr-view" }, [this.view2d, this.remote]),
        el("div", { className: "ihmr-view" }, [this.canvas3d]),
      ]),
      this.status,
    ]);
    for (const input of [this.model, this.detector, this.source, this.fov, this.every, this.vmc]) {
      input.addEventListener("change", () => this.saveSettings());
    }
    this.character.addEventListener("change", () => {
      this.saveSettings();
      this.loadCharacter();
    });
    this.startBtn.onclick = () => (this.ws ? this.stop() : this.start());
    this.recBtn.onclick = () => this.toggleRecording();
    this.fillModels();
  }

  async fillModels() {
    const [models, detectors, characters] = await Promise.all([
      listModels("detection", /instanthmr/i), listModels("diffusion_models", /rt_detr/i), (await api.fetchApi("/instanthmr/characters")).json(),
    ]);
    this.model.replaceChildren(...models.map((m) => el("option", { value: m, textContent: m })));
    this.detector.replaceChildren(...detectors.map((m) => el("option", { value: m, textContent: m })));
    this.character.replaceChildren(el("option", { value: "", textContent: "MHR body" }), ...characters.map((c) => el("option", { value: c, textContent: c })));
    if (models.includes(this.settings.model)) this.model.value = this.settings.model;
    if (detectors.includes(this.settings.detector)) this.detector.value = this.settings.detector;
    if (characters.includes(this.settings.character)) this.character.value = this.settings.character;
    this.saveSettings();
  }

  saveSettings() {
    Object.assign(this.settings, {
      model: this.model.value, detector: this.detector.value, source: this.source.value,
      fov: Number(this.fov.value) || 0, detect_every: Math.max(1, Number(this.every.value) || 1),
      character: this.character.value, vmc: this.vmc.value.trim(),
    });
    if (this.ws?.readyState === WebSocket.OPEN) {
      const { model, detector, fov, detect_every, vmc } = this.settings;
      this.ws.send(JSON.stringify({ type: "settings", model, detector, fov, detect_every, vmc }));
    }
  }

  async loadCharacter() {
    this.characters.forEach((c) => c.dispose());
    this.characters = [];
    this.characterSource = null;
    const file = this.settings.character;
    if (!file || !this.renderer) return;
    const slash = file.lastIndexOf("/");
    const subfolder = slash < 0 ? "" : file.slice(0, slash);
    const url = api.apiURL(`/view?type=input&subfolder=${encodeURIComponent(subfolder)}&filename=${encodeURIComponent(file.slice(slash + 1))}`);
    this.status.textContent = `Loading ${file}...`;
    try {
      const { loadCharacter, Character } = await import("/instanthmr/static/character.js");
      const source = { ...(await loadCharacter(url, file)), Character };
      if (this.settings.character === file) this.characterSource = source;
      this.status.textContent = `Loaded ${file}`;
    } catch (e) {
      this.status.textContent = `${file}: ${e.message}`;
    }
  }

  async start() {
    await this.initThree();
    if (this.settings.character && !this.characterSource) this.loadCharacter();
    const ws = (this.ws = new WebSocket(liveUrl(this.settings.room, "viewer")));
    ws.binaryType = "arraybuffer";
    ws.onopen = () => this.saveSettings();
    ws.onmessage = (e) => (typeof e.data === "string" ? this.onJson(JSON.parse(e.data)) : this.onBinary(e.data));
    ws.onclose = () => this.ws === ws && this.stop();
    this.startBtn.textContent = "Stop";
    this.recBtn.disabled = false;
    this.framed = false;
    this.lastFrame = performance.now();
    if (this.settings.source === "local") {
      this.remote.replaceChildren();
      this.camera = new CameraStreamer(this.settings.room, this.video, () => {});
      try {
        await this.camera.start("user");
      } catch (e) {
        this.status.textContent = e.message;
      }
    } else {
      const url = await phoneUrl(this.settings.room);
      const { default: qrcode } = await import("/instanthmr/static/lib/qrcode.mjs");
      const qr = qrcode(0, "M");
      qr.addData(url);
      qr.make();
      const hint = location.protocol === "https:" ? "Scan with the phone, accept the certificate warning, then tap Start camera"
        : "Phones need HTTPS for the camera: start ComfyUI with --tls-keyfile and --tls-certfile";
      this.remote.replaceChildren(el("img", { src: qr.createDataURL(4, 0) }), el("a", { href: url, target: "_blank", textContent: url }), el("div", { textContent: hint }));
      this.status.textContent = "Waiting for a camera";
    }
  }

  stop() {
    this.camera?.stop();
    this.camera = null;
    const ws = this.ws;
    this.ws = null;
    ws?.close();
    this.startBtn.textContent = "Start";
    this.recBtn.disabled = true;
    this.recBtn.classList.remove("on");
    this.recording = false;
    this.status.textContent = "Stopped";
  }

  toggleRecording() {
    this.recording = !this.recording;
    this.recBtn.classList.toggle("on", this.recording);
    this.recBtn.textContent = this.recording ? "■ Stop" : "● Rec";
    this.ws?.send(JSON.stringify({ type: "record", on: this.recording }));
  }

  async onJson(msg) {
    if (msg.type === "error") {
      this.status.textContent = msg.error;
    } else if (msg.type === "cameras") {
      this.cameras = msg.count;
      if (msg.count) this.remote.replaceChildren();
    } else if (msg.type === "recorded") {
      const widget = this.node.widgets.find((w) => w.name === "recording");
      widget.options.values = await (await api.fetchApi("/instanthmr/recordings")).json();
      widget.value = msg.file;
      widget.callback?.(msg.file);
      this.saved = `saved ${msg.file} (${msg.frames} frames, ${msg.fps.toFixed(1)} fps)`;
      this.status.textContent = this.saved;
    }
  }

  async onBinary(buf) {
    const headLen = new DataView(buf).getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headLen)));
    let offset = 4 + headLen;
    const parts = header.sizes.map((size) => {
      const part = buf.slice(offset, offset + size);
      offset += size;
      return part;
    });
    if (header.type === "faces") {
      this.faces = new Uint32Array(new Int32Array(parts[0]));
      this.boneTable = header.bones;
      this.meshes.forEach((m) => m.geometry.setIndex(null));
      return;
    }
    const bitmap = await createImageBitmap(new Blob([parts[0]], { type: "image/jpeg" }));
    this.draw2d(bitmap, header);
    this.update3d(new Float32Array(parts[1]), header.persons);
    const now = performance.now();
    const fps = 1000 / (now - this.lastFrame);
    this.lastFrame = now;
    if (!header.error) {
      this.status.textContent = `${fps.toFixed(1)} fps · detector ${header.det_ms.toFixed(0)} ms · InstantHMR ${header.hmr_ms.toFixed(0)} ms · ${header.persons.length} people`
        + (header.recorded ? ` · REC ${header.recorded} frames` : this.saved ? ` · ${this.saved}` : "");
    }
  }

  draw2d(bitmap, header) {
    const c = this.view2d;
    const r = c.getBoundingClientRect();
    c.width = Math.max(1, Math.round(r.width));
    c.height = Math.max(1, Math.round(r.height));
    const scale = Math.min(c.width / bitmap.width, c.height / bitmap.height);
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.save();
    ctx.translate((c.width - bitmap.width * scale) / 2, (c.height - bitmap.height * scale) / 2);
    ctx.drawImage(bitmap, 0, 0, bitmap.width * scale, bitmap.height * scale);
    drawPersons(ctx, header.persons, scale);
    ctx.restore();
    bitmap.close();
  }

  async initThree() {
    if (this.renderer) return;
    const THREE = await import("/instanthmr/static/lib/three.module.min.js");
    const { OrbitControls } = await import("/instanthmr/static/lib/OrbitControls.js");
    this.THREE = THREE;
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas3d, antialias: true });
    this.renderer.setClearColor(0x0b0b0b);
    this.scene = new THREE.Scene();
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x333344, 1.2));
    const sun = new THREE.DirectionalLight(0xffffff, 1.5);
    sun.position.set(1, 2, 3);
    this.scene.add(sun);
    // Camera space of the input: the capturing camera sits at the origin and looks down -Z.
    this.scene.add(new THREE.AxesHelper(0.3));
    this.camera3d = new THREE.PerspectiveCamera(45, 1, 0.05, 100);
    this.camera3d.position.set(2.5, 1.5, 1.5);
    this.controls = new OrbitControls(this.camera3d, this.canvas3d);
    this.controls.target.set(0, -0.5, -4);
    this.controls.update();
    this.material = new THREE.MeshStandardMaterial({ color: 0xd8c3a5, roughness: 0.7 });
    this.resize = new ResizeObserver(() => this.render3d());
    this.resize.observe(this.canvas3d);
    this.controls.addEventListener("change", () => this.render3d());
  }

  updateCharacters(persons) {
    try {
      while (this.characters.length < persons.length) {
        const character = new this.characterSource.Character(this.characterSource, this.boneTable);
        this.scene.add(character.root);
        this.characters.push(character);
      }
    } catch (e) {
      this.characterSource = null;
      this.status.textContent = `${this.settings.character}: ${e.message}`;
      return;
    }
    persons.forEach((p, i) => this.characters[i].apply(p));
  }

  update3d(verts, persons) {
    const THREE = this.THREE;
    const count = persons.length;
    const n = 18439 * 3;
    while (this.meshes.length < count) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n), 3));
      const mesh = new THREE.Mesh(geometry, this.material);
      mesh.frustumCulled = false;
      this.scene.add(mesh);
      this.meshes.push(mesh);
    }
    const character = this.characterSource && this.boneTable;
    if (character) this.updateCharacters(persons);
    this.characters.forEach((c, i) => (c.root.visible = character && i < count));
    this.meshes.forEach((mesh, i) => {
      mesh.visible = !character && i < count;
      if (i >= count) return;
      if (!mesh.geometry.index && this.faces) mesh.geometry.setIndex(new THREE.BufferAttribute(this.faces, 1));
      const pos = mesh.geometry.attributes.position.array;
      for (let j = 0; j < n; j += 3) {
        pos[j] = verts[i * n + j];
        pos[j + 1] = -verts[i * n + j + 1];
        pos[j + 2] = -verts[i * n + j + 2];
      }
      mesh.geometry.attributes.position.needsUpdate = true;
      mesh.geometry.computeVertexNormals();
    });
    if (count && !this.framed) {
      // Frame the first person once, from slightly off the capturing camera's axis.
      this.framed = true;
      this.meshes[0].geometry.computeBoundingSphere();
      const { center, radius } = this.meshes[0].geometry.boundingSphere;
      this.controls.target.copy(center);
      this.camera3d.position.set(center.x + radius * 1.2, center.y + radius * 0.4, center.z + radius * 3);
      this.controls.update();
    }
    this.render3d();
  }

  render3d() {
    if (!this.renderer) return;
    const r = this.canvas3d.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(w, h, false);
    this.camera3d.aspect = w / h;
    this.camera3d.updateProjectionMatrix();
    this.renderer.render(this.scene, this.camera3d);
  }

  dispose() {
    this.stop();
    this.resize?.disconnect();
    this.controls?.dispose();
    this.meshes.forEach((m) => m.geometry.dispose());
    this.material?.dispose();
    this.renderer?.dispose();
  }
}

app.registerExtension({
  name: "InstantHMR.LiveCapture",
  init() {
    document.head.append(el("style", { textContent: STYLE }));
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "InstantHMRLiveCapture") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      const node = this;
      const container = el("div", { style: "width: 100%; height: 100%;" });
      node.addDOMWidget("live_view", "instanthmr_live", container, { serialize: false, getMinHeight: () => 420 });
      node.setSize([760, 560]);
      // The saved room and settings arrive with configure(), after onNodeCreated.
      queueMicrotask(() => {
        node.instanthmrLive = new LiveView(node);
        container.append(node.instanthmrLive.root);
      });
      const onRemoved = node.onRemoved;
      node.onRemoved = function () {
        node.instanthmrLive?.dispose();
        return onRemoved?.apply(this, arguments);
      };
    };
  },
});
