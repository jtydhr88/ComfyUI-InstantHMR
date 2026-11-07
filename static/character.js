// Drives a rigged character (Mixamo / Unreal / VRM style FBX, glTF or VRM) with the humanoid bone rotations of the live view.
import * as THREE from "./lib/three.module.min.js";
import { FBXLoader } from "./lib/FBXLoader.js";
import { GLTFLoader } from "./lib/GLTFLoader.js";
import { clone } from "./lib/SkeletonUtils.js";

const VRM1_THUMB = { Metacarpal: "Proximal", Proximal: "Intermediate", Distal: "Distal" };

// Checked in order: fingers before the hand, forearm before arm, thigh before leg.
const PARTS = [
  ["Thumb", /thumb/], ["Index", /index/], ["Middle", /middle/], ["Ring", /ring/], ["Little", /pinky|little/],
  ["Toes", /toe|ball/], ["Foot", /foot|ankle/], ["UpperLeg", /upleg|thigh|upperleg/], ["LowerLeg", /leg|calf|shin|knee/],
  ["Shoulder", /shoulder|clavicle|collar/], ["LowerArm", /forearm|lowerarm|elbow/], ["UpperArm", /arm/], ["Hand", /hand|wrist/],
  ["Neck", /neck/], ["Head", /head/], ["Hips", /hips|pelvis/],
];

function side(name) {
  const n = name.toLowerCase();
  if (/left|(^|[^a-z])l([^a-z]|$)/.test(n) || /[a-z0-9]L$/.test(name)) return "Left";
  if (/right|(^|[^a-z])r([^a-z]|$)/.test(n) || /[a-z0-9]R$/.test(name)) return "Right";
  return "";
}

function depth(o) {
  let d = 0;
  for (; o.parent; o = o.parent) d++;
  return d;
}

/** Humanoid bone name -> object, from VRM metadata or from bone names. */
function findBones(root, vrmNames) {
  if (vrmNames) {
    return Object.fromEntries(Object.entries(vrmNames).map(([k, name]) => [k, root.getObjectByName(name)]).filter(([, o]) => o));
  }
  const bones = [];
  root.traverse((o) => o.isBone && bones.push(o));
  const found = {};
  const groups = {};
  for (const b of bones) {
    const n = b.name.toLowerCase().replace(/mixamorig\d*|cc_base_|bip0*1/g, "");
    if (/twist|roll|metacarpal(?!.*thumb)|end$|_end|nub|tip/.test(n)) continue;
    const part = PARTS.find(([, re]) => re.test(n))?.[0];
    if (!part) continue;
    const s = part === "Hips" || part === "Neck" || part === "Head" ? "" : side(b.name);
    if (part !== "Hips" && part !== "Neck" && part !== "Head" && !s) continue;
    (groups[s + part] ??= []).push(b);
  }
  for (const [key, list] of Object.entries(groups)) {
    list.sort((a, b) => depth(a) - depth(b));
    if (/Thumb|Index|Middle|Ring|Little/.test(key)) {
      ["Proximal", "Intermediate", "Distal"].forEach((seg, i) => list[i] && (found[key + seg] = list[i]));
    } else {
      found[key] = list[0];
    }
  }
  // Spine bones are whatever sits between the hips and the neck.
  if (found.Hips && found.Neck) {
    const chain = [];
    for (let o = found.Neck.parent; o && o !== found.Hips; o = o.parent) chain.unshift(o);
    if (chain.length) found.Spine = chain[0];
    if (chain.length >= 2) found.Chest = chain[chain.length >= 3 ? Math.floor((chain.length - 1) / 2) : 1];
    if (chain.length >= 3) found.UpperChest = chain[chain.length - 1];
  }
  return found;
}

async function vrmBoneNames(gltf) {
  const ext = gltf.parser.json.extensions ?? {};
  let list;
  if (ext.VRMC_vrm) {
    list = Object.entries(ext.VRMC_vrm.humanoid.humanBones).map(([bone, v]) => [bone.replace(/Thumb(Metacarpal|Proximal|Distal)$/, (_, s) => `Thumb${VRM1_THUMB[s]}`), v.node]);
  } else if (ext.VRM) {
    list = ext.VRM.humanoid.humanBones.map((b) => [b.bone, b.node]);
  } else {
    return null;
  }
  const names = {};
  for (const [bone, node] of list) names[bone[0].toUpperCase() + bone.slice(1)] = (await gltf.parser.getDependency("node", node)).name;
  return names;
}

export async function loadCharacter(url, file) {
  if (file.toLowerCase().endsWith(".fbx")) {
    return { object: await new FBXLoader().loadAsync(url), vrmNames: null };
  }
  const gltf = await new GLTFLoader().loadAsync(url);
  return { object: gltf.scene, vrmNames: await vrmBoneNames(gltf) };
}

const swing = (from, to) => new THREE.Quaternion().setFromUnitVectors(from, to);

export class Character {
  /** source: loadCharacter() result; table: [[name, parent, child, aim]] from the server, parents first. */
  constructor(source, table) {
    this.object = clone(source.object);
    this.object.traverse((o) => o.isMesh && (o.frustumCulled = false));
    this.root = new THREE.Group();
    this.root.add(this.object);
    this.table = table;
    const bones = findBones(this.object, source.vrmNames);
    this.bones = table.map(([name]) => bones[name] ?? null);
    if (!bones.Hips || !bones.LeftUpperArm || !bones.RightUpperArm || !bones.Head) {
      throw new Error(`Could not find the hips, head and arms in this character (found: ${Object.keys(bones).join(", ") || "no bones"})`);
    }
    // Turn the character so it stands like the MHR body: Y up, facing +Z, its left side at +X.
    const at = (o) => o.getWorldPosition(new THREE.Vector3());
    this.root.updateMatrixWorld(true);
    const x = at(bones.LeftUpperArm).sub(at(bones.RightUpperArm)).normalize();
    const up = at(bones.Head).sub(at(bones.Hips));
    const y = up.sub(x.clone().multiplyScalar(up.dot(x))).normalize();
    const basis = new THREE.Matrix4().makeBasis(x, y, new THREE.Vector3().crossVectors(x, y));
    this.root.quaternion.setFromRotationMatrix(basis).invert();
    this.root.updateMatrixWorld(true);
    // T-pose orientation of each bone: rest orientation swung onto the T-pose direction, or kept relative to the parent.
    const index = Object.fromEntries(table.map(([name], i) => [name, i]));
    const turn = [];
    this.tpose = table.map(([, parent, child, aim], i) => {
      const bone = this.bones[i];
      const childBone = child && this.bones[index[child]];
      if (aim && bone && childBone) {
        turn[i] = swing(at(childBone).sub(at(bone)).normalize(), new THREE.Vector3(...aim));
      } else {
        turn[i] = parent ? turn[index[parent]] : new THREE.Quaternion();
      }
      return bone ? turn[i].clone().multiply(bone.getWorldQuaternion(new THREE.Quaternion())) : null;
    });
    const L = (n) => at(bones[n] ?? bones.Hips);
    this.leg = L("LeftUpperLeg").distanceTo(L("LeftLowerLeg")) + L("LeftLowerLeg").distanceTo(L("LeftFoot")) || 1;
    this.hips = bones.Hips;
  }

  /** person: { bones: [x, y, z, w] per table bone (world, MHR frame), hips: [x, y, z], leg: metres }. */
  apply(person) {
    const world = new THREE.Quaternion();
    const parent = new THREE.Quaternion();
    this.root.position.set(0, 0, 0);
    this.root.scale.setScalar(person.leg / this.leg);
    this.root.updateMatrixWorld(true);
    this.bones.forEach((bone, i) => {
      if (!bone) return;
      world.fromArray(person.bones, i * 4).multiply(this.tpose[i]);
      bone.parent.updateWorldMatrix(true, false);
      bone.parent.getWorldQuaternion(parent);
      bone.quaternion.copy(parent.invert().multiply(world));
      bone.updateMatrixWorld(true);
    });
    this.root.position.add(new THREE.Vector3(...person.hips).sub(this.hips.getWorldPosition(new THREE.Vector3())));
    this.root.updateMatrixWorld(true);
  }

  dispose() {
    this.root.removeFromParent();
  }
}
