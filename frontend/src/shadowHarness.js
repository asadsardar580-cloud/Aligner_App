// shadowHarness.js — a real WebGL reproduction of the viewport's lighting rig,
// built so the shadow subsystem can be A/B tested against MEASURED PIXELS.
//
// WHY THIS EXISTS. `verify-shadowrig.mjs` asserts that every number handed to a
// light is finite, correctly signed and inside its own frustum. It cannot tell
// you the viewport is lit, and it did not: the rig computed a perfectly finite
// answer while the screen went black, because the defect was a plane SIZED
// against nothing (CLAUDE.md 20.2). Arithmetic that is right and a picture that
// is wrong is the exact shape of this bug, twice now, so this harness renders
// and reads the framebuffer back.
//
// It is NOT the app. It is the app's rig — same materials, same tone mapping,
// same PMREM environment, same sun and catcher construction, same
// `computeShadowRig` — over a synthetic arch, so it needs no backend, no scan
// and no patient data, and it can run in CI.
//
// THE MEASUREMENT IS MASKED, and that is the point. "The cast looks dark" and
// "the frame looks dark" are different claims with different causes, and a mean
// over the whole image cannot separate them. An ID pass renders the arch white
// on black with the catcher hidden, giving a per-pixel mask of exactly the cast;
// every luminance below is averaged over THAT mask, so a dark wash behind the
// cast cannot be mistaken for the cast going dark, nor the other way round.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { computeShadowRig } from "./shadowRig.js";

const CATCHER_BASE_MM = 400;
const W = 640, H = 480;

/** A synthetic mandibular cast: a horseshoe band with discrete crowns on top.
 *  Anatomy enough for the rig to be asked a real question; no patient data. */
function buildArch() {
  const pos = [], idx = [];
  const RINGS = 14, STEPS = 96;
  const R = 22, halfW = 7, height = 16;
  // Horseshoe from -140deg to +140deg, so it opens posteriorly like an arch.
  for (let s = 0; s <= STEPS; s++) {
    const a = (-140 + (280 * s) / STEPS) * Math.PI / 180;
    for (let r = 0; r <= RINGS; r++) {
      const t = r / RINGS;
      // Cross-section: a rounded ridge over a vertical wall.
      const across = (t - 0.5) * 2 * halfW;
      const ridge = Math.cos((t - 0.5) * Math.PI) ** 2;
      // Crowns: a periodic bump along the arch, taller at the front.
      const tooth = 0.5 + 0.5 * Math.cos(a * 7);
      const z = -height + ridge * (6 + 3 * tooth);
      pos.push((R + across) * Math.cos(a), (R + across) * Math.sin(a), z);
    }
  }
  const at = (s, r) => s * (RINGS + 1) + r;
  for (let s = 0; s < STEPS; s++) {
    for (let r = 0; r < RINGS; r++) {
      idx.push(at(s, r), at(s + 1, r), at(s + 1, r + 1));
      idx.push(at(s, r), at(s + 1, r + 1), at(s, r + 1));
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  const col = new Float32Array(pos.length).fill(0.82);
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  g.computeBoundingBox();
  g.computeBoundingSphere();
  return g;
}

/** The app's occlusal basis for this synthetic arch: it lies in the XY plane
 *  with the occlusal direction along +Z, anterior along +Y. */
const FRAME = { u_occ: [0, 0, 1], u_sag: [0, 1, 0], u_tra: [1, 0, 0] };

export function boot(mount) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x101418);

  const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 5000);
  const renderer = new THREE.WebGLRenderer({
    antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(W, H);
  renderer.setPixelRatio(1);                    // 1:1 pixels, so readback maps
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  mount.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = false;

  // --- the app's camera-parented three-point rig, verbatim ---------------
  const key = new THREE.DirectionalLight(0xfff6ec, 2.1);
  key.position.set(-1, 1, 1);
  const fill = new THREE.DirectionalLight(0xdce9ff, 0.55);
  fill.position.set(1.4, -0.6, 0.8);
  const rim = new THREE.DirectionalLight(0xffffff, 0.9);
  rim.position.set(0.2, 1.1, -1.4);
  camera.add(key, fill, rim);
  scene.add(camera);
  const ambient = new THREE.AmbientLight(0xffffff, 0.18);
  scene.add(ambient);

  // --- the sun and catcher, verbatim ------------------------------------
  const sun = new THREE.DirectionalLight(0xffffff, 0.0);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 400;
  sun.shadow.bias = -0.0015;
  scene.add(sun, sun.target);

  const catcher = new THREE.Mesh(
    new THREE.PlaneGeometry(CATCHER_BASE_MM, CATCHER_BASE_MM),
    new THREE.ShadowMaterial({ opacity: 0.22 }));
  catcher.receiveShadow = true;
  catcher.visible = false;
  scene.add(catcher);

  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.shadowMap.autoUpdate = false;

  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.45;
  pmrem.dispose();

  // --- the cast ----------------------------------------------------------
  const geom = buildArch();
  const material = new THREE.MeshPhysicalMaterial({
    vertexColors: true, roughness: 0.35, metalness: 0.05,
    clearcoat: 0.6, clearcoatRoughness: 0.25, side: THREE.DoubleSide });
  const arch = new THREE.Mesh(geom, material);
  scene.add(arch);

  // Frame the cast the way frameArch does: off the occlusal axis, anterior down.
  const bs = geom.boundingSphere;
  const up = new THREE.Vector3(0, 0, 1);
  const inPlane = new THREE.Vector3(1, 0, 0);
  const tilt = 0.57;                            // ~34.7 degrees, as measured
  const dir = up.clone().multiplyScalar(Math.sqrt(1 - tilt * tilt))
    .add(inPlane.clone().multiplyScalar(tilt)).normalize();
  const halfV = THREE.MathUtils.degToRad(camera.fov) / 2;
  const halfH = Math.atan(Math.tan(halfV) * camera.aspect);
  const dist = bs.radius / Math.sin(Math.min(halfV, halfH));
  camera.up.set(0, -1, 0);
  camera.position.copy(bs.center).add(dir.multiplyScalar(dist));
  camera.near = dist / 100;
  camera.far = dist * 4 + bs.radius * 4;
  camera.updateProjectionMatrix();
  controls.target.copy(bs.center);
  controls.update();

  const render = () => {
    renderer.shadowMap.needsUpdate = true;
    renderer.render(scene, camera);
  };

  /** aimShadows, reduced to the part under test. `opts` turns individual
   *  components on and off so each can be blamed or cleared independently. */
  function aim(opts = {}) {
    const {
      enableSun = true, enableCatcher = true, castShadow = true,
      shadowMapEnabled = true, intensity = null,
    } = opts;

    const box = new THREE.Box3().setFromObject(arch);
    const rig = computeShadowRig({
      box: { min: box.min.toArray(), max: box.max.toArray() }, ...FRAME });
    if (!rig) throw new Error("computeShadowRig returned null");

    catcher.position.fromArray(rig.catcherPosition);
    catcher.quaternion.fromArray(rig.catcherQuaternion);
    catcher.scale.setScalar(rig.catcherSize / CATCHER_BASE_MM);
    catcher.visible = enableCatcher;

    sun.target.position.fromArray(rig.targetPosition);
    sun.position.fromArray(rig.sunPosition);
    sun.intensity = intensity !== null ? intensity : (enableSun ? rig.intensity : 0);
    sun.castShadow = castShadow;
    Object.assign(sun.shadow.camera, rig.frustum);
    sun.shadow.camera.updateProjectionMatrix();
    renderer.shadowMap.enabled = shadowMapEnabled;
    arch.castShadow = true;

    scene.updateMatrixWorld(true);
    sun.target.updateMatrixWorld(true);
    render();
    return rig;
  }

  /** Per-pixel mask of exactly the cast: the arch white on black, no catcher,
   *  no lighting model. This is what every luminance below is averaged over. */
  function castMask() {
    const keep = { vis: catcher.visible, env: scene.environment,
                   bg: scene.background, tone: renderer.toneMapping };
    catcher.visible = false;
    scene.environment = null;
    scene.background = new THREE.Color(0x000000);
    renderer.toneMapping = THREE.NoToneMapping;
    // DoubleSide, like the cast material it stands in for. FrontSide culls
    // the horseshoe's inward-facing band and the mask collapsed to 238 px.
    const flat = new THREE.MeshBasicMaterial({
      color: 0xffffff, side: THREE.DoubleSide });
    arch.material = flat;
    renderer.render(scene, camera);
    const buf = new Uint8Array(W * H * 4);
    renderer.getContext().readPixels(
      0, 0, W, H, renderer.getContext().RGBA,
      renderer.getContext().UNSIGNED_BYTE, buf);
    arch.material = material;
    catcher.visible = keep.vis;
    scene.environment = keep.env;
    scene.background = keep.bg;
    renderer.toneMapping = keep.tone;
    flat.dispose();
    const mask = new Uint8Array(W * H);
    let n = 0;
    for (let i = 0; i < W * H; i++) {
      // Anti-aliased edge pixels are part background; require a solid hit so
      // the mask cannot drift onto the catcher behind the silhouette.
      if (buf[i * 4] > 200) { mask[i] = 1; n++; }
    }
    return { mask, count: n };
  }

  /** Mean luminance over the mask, and over everything outside it. */
  function measure(mask) {
    const buf = new Uint8Array(W * H * 4);
    const gl = renderer.getContext();
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    let onSum = 0, onN = 0, offSum = 0, offN = 0, onDark = 0;
    for (let i = 0; i < W * H; i++) {
      const L = 0.2126 * buf[i * 4] + 0.7152 * buf[i * 4 + 1] + 0.0722 * buf[i * 4 + 2];
      if (mask[i]) { onSum += L; onN++; if (L < 40) onDark++; }
      else { offSum += L; offN++; }
    }
    return {
      cast: onN ? onSum / onN : 0,
      background: offN ? offSum / offN : 0,
      castPixels: onN,
      castDarkFraction: onN ? onDark / onN : 0,
    };
  }

  /** Where the catcher's own corners land in SHADOW-CAMERA space, which is the
   *  question `verify-shadowrig.mjs` never asks. A corner outside [-1,1] after
   *  the shadow camera's projection samples the depth texture with clamped UVs
   *  and comes back SHADOWED whatever is really there. */
  function catcherInShadowCamera() {
    sun.shadow.camera.updateMatrixWorld(true);
    sun.shadow.camera.updateProjectionMatrix();
    const m = new THREE.Matrix4().multiplyMatrices(
      sun.shadow.camera.projectionMatrix,
      sun.shadow.camera.matrixWorldInverse);
    catcher.updateMatrixWorld(true);
    const out = [];
    const half = CATCHER_BASE_MM / 2;
    for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const p = new THREE.Vector3(sx * half, sy * half, 0)
        .applyMatrix4(catcher.matrixWorld).applyMatrix4(m);
      out.push([p.x, p.y, p.z]);
    }
    const outside = out.filter(
      (p) => Math.abs(p[0]) > 1 || Math.abs(p[1]) > 1).length;
    return { corners: out, cornersOutsideShadowMap: outside };
  }

  /** Is the catcher between the camera and the cast? It must never be. */
  function catcherOrdering() {
    catcher.updateMatrixWorld(true);
    const camPos = camera.position.clone();
    const castC = geom.boundingSphere.center.clone();
    const catC = catcher.position.clone();
    const n = new THREE.Vector3(0, 0, 1).applyQuaternion(catcher.quaternion);
    return {
      cameraToCast: camPos.distanceTo(castC),
      cameraToCatcher: camPos.distanceTo(catC),
      catcherIsInFront: camPos.distanceTo(catC) < camPos.distanceTo(castC),
      catcherNormal: n.toArray(),
      catcherFacesCamera: n.dot(camPos.clone().sub(catC).normalize()) > 0,
      catcherPosition: catC.toArray(),
      castCentre: castC.toArray(),
      castRadius: geom.boundingSphere.radius,
    };
  }

  return { scene, camera, renderer, sun, catcher, arch, aim, render,
           castMask, measure, catcherInShadowCamera, catcherOrdering,
           rigFor: () => computeShadowRig({
             box: { min: new THREE.Box3().setFromObject(arch).min.toArray(),
                    max: new THREE.Box3().setFromObject(arch).max.toArray() },
             ...FRAME }) };
}

if (typeof window !== "undefined") {
  const mount = document.getElementById("app") || document.body;
  window.__harness = boot(mount);
  window.__harnessReady = true;
}
