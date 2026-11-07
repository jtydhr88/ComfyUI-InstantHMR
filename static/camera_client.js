// Shared by the phone camera page and the Live Capture node.

const MAX_SIDE = 960;
// Frames sent before the first result comes back: the next frame is encoded and on the wire while the server works.
const IN_FLIGHT = 2;

const FINGERS = (wrist, first) => [0, 4, 8, 12, 16].flatMap((f) => {
  const [tip, j1, j2, j3] = [first + f, first + f + 1, first + f + 2, first + f + 3];
  return [[wrist, j3], [j3, j2], [j2, j1], [j1, tip]];
});

// MHR70 keypoints
export const EDGES = [
  [0, 69], [69, 5], [69, 6], [5, 9], [6, 10], [9, 10],
  [5, 7], [7, 62], [6, 8], [8, 41],
  [9, 11], [11, 13], [13, 17], [13, 15], [10, 12], [12, 14], [14, 20], [14, 18],
  ...FINGERS(41, 21), ...FINGERS(62, 42),
];

const COLORS = ["#4ade80", "#f472b6", "#60a5fa", "#facc15"];

export function liveUrl(room, role) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/instanthmr/live?room=${room}&role=${role}`;
}

export function drawPersons(ctx, persons, scale) {
  ctx.lineWidth = 2;
  persons.forEach((p, i) => {
    const color = COLORS[i % COLORS.length];
    const [x1, y1, x2, y2] = p.bbox.map((v) => v * scale);
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.5;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.globalAlpha = 1;
    const k = p.kp2d;
    ctx.beginPath();
    for (const [a, b] of EDGES) {
      ctx.moveTo(k[2 * a] * scale, k[2 * a + 1] * scale);
      ctx.lineTo(k[2 * b] * scale, k[2 * b + 1] * scale);
    }
    ctx.stroke();
  });
}

export class CameraStreamer {
  constructor(room, video, onResult) {
    this.room = room;
    this.video = video;
    this.onResult = onResult;
    this.canvas = document.createElement("canvas");
  }

  async start(facingMode) {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("The camera needs HTTPS. Start ComfyUI with --tls-keyfile and --tls-certfile, or open it on localhost.");
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    const ws = (this.ws = new WebSocket(liveUrl(this.room, "camera")));
    ws.binaryType = "arraybuffer";
    ws.onopen = async () => {
      for (let i = 0; i < IN_FLIGHT; i++) await this.sendFrame();
    };
    ws.onmessage = (e) => {
      this.onResult(JSON.parse(e.data));
      this.sendFrame();
    };
    ws.onclose = () => this.ws === ws && this.stop();
  }

  async sendFrame() {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    const v = this.video;
    const s = Math.min(1, MAX_SIDE / Math.max(v.videoWidth, v.videoHeight));
    this.canvas.width = Math.round(v.videoWidth * s);
    this.canvas.height = Math.round(v.videoHeight * s);
    this.canvas.getContext("2d").drawImage(v, 0, 0, this.canvas.width, this.canvas.height);
    const blob = await new Promise((r) => this.canvas.toBlob(r, "image/jpeg", 0.8));
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(await blob.arrayBuffer());
  }

  stop() {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    const ws = this.ws;
    this.ws = null;
    ws?.close();
  }
}
