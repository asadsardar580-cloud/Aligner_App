import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { frameArch, attachResize, pickAcrossArches } from "./frameArch";
import { BrushIndex, installBrush, CELL_FACTOR } from "./brush";
import { installBVH, refreshBoundsTree, dropBoundsTree, configureRaycaster } from "./bvh";
import { ToothGizmo, pickOcclusalPlane, rootDefaultForFDI, ROOT_DEFAULTS_MM,
         toothAxes, deltaFromClinical, clinicalAtStage, stagingFor } from "./toothGizmo";
import StagingTimeline from "./StagingTimeline";
import { useStagePlayback } from "./useStagePlayback";
import ValidationPanel, { PASS, REVIEW, UNKNOWN } from "./ValidationPanel";
import AttachmentPanel from "./AttachmentPanel";
import { installAttachmentTool, SHAPES as ATTACHMENT_SHAPES } from "./AttachmentPlacementTool";
import { PanelGroup, Panel } from "./Panel";

/** Which Wheeler class an FDI number belongs to, for the label next to the slider. */
function fdiClass(fdi) {
  const pos = Number(String(fdi).slice(-1));
  if (pos <= 2) return "incisor";
  if (pos === 3) return "canine";
  if (pos <= 5) return "premolar";
  return "molar";
}

const API = "http://127.0.0.1:8000";

// Where the per-arch session ids are kept so a refresh can find the case again.
// Ids only — the scan itself never leaves the server's memory.
const CASE_KEY = "aligner.case.sessions";

// Backend connection states. "offline" is the one that matters: without a
// permanent indicator, an API that was simply never started is indistinguishable
// from a broken app — you find out only when an upload fails.
const HEALTH_UI = {
  checking:  { dot: "#8b93a0", label: "Checking backend…" },
  connected: { dot: "#3cb44b", label: "Backend connected" },
  warming:   { dot: "#ffe119", label: "Backend up, AI loading" },
  degraded:  { dot: "#f58231", label: "Backend up, AI unavailable" },
  offline:   { dot: "#e6194b", label: "Backend not running" },
};
const GINGIVA = new THREE.Color("#d9a7a0");
const SELECTED = new THREE.Color("#3fa9ff");
const CROWN_COL = new THREE.Color("#2ecc71");

/**
 * Release GPU memory for a mesh and everything under it.
 *
 * Three does not free buffers when an object leaves the scene graph — dropping
 * the reference alone leaks the VBOs. Materials leak separately from geometry,
 * which is the half that was missing: loadArch disposed the geometry and left
 * the material behind on every reload.
 */
function disposeMesh(obj) {
  if (!obj) return;
  obj.traverse?.((n) => {
    // The bounds tree is a separate allocation from the geometry buffers and
    // is not freed by geometry.dispose(). Leaking one per loaded arch is how a
    // long session ends up holding several hundred MB of stale acceleration
    // structure.
    dropBoundsTree(n.geometry);
    n.geometry?.dispose?.();
    const m = n.material;
    if (Array.isArray(m)) m.forEach((x) => x?.dispose?.());
    else m?.dispose?.();
  });
  obj.parent?.remove(obj);
}

/**
 * Enamel/tissue material — MeshPhysicalMaterial, not Phong.
 *
 * clearcoat is a PHYSICAL-material property; MeshStandardMaterial does not
 * have it. Physical extends Standard, so roughness/metalness behave exactly as
 * specified and the thin specular coat on top is what reads as glazed ceramic
 * rather than shiny plastic. The values are the dental-CAD convention: enamel
 * is a dielectric, so metalness stays near zero and the highlight comes from
 * the coat.
 *
 * side stays DoubleSide: an arch mid-extraction has open boundaries at every
 * socket, and backface-culling them would punch visible holes through the cast.
 */
const tissueMaterial = (over = {}) => new THREE.MeshPhysicalMaterial({
  vertexColors: true,
  roughness: 0.35,
  metalness: 0.05,
  clearcoat: 0.6,
  clearcoatRoughness: 0.25,
  side: THREE.DoubleSide,
  ...over,
});

/**
 * The virtual root — drawn so it can NEVER be mistaken for scan data.
 *
 * There is no root in an intraoral scan; it stops at the gingival margin. This
 * cone is an extrapolation from two numbers, the cervical rim and a Wheeler
 * average, and the same length drives C_res — so a clinician looking at it is
 * looking at the assumption their whole prescription pivots about.
 *
 * Wireframe, 28% opaque, in a violet that appears nowhere else in the scene
 * (tissue is pink, crowns are green, selection is blue). depthWrite off so it
 * never occludes real geometry, and it is kept out of every pick list.
 */
const rootMaterial = () => new THREE.MeshBasicMaterial({
  color: 0x9d7bff,
  wireframe: true,
  transparent: true,
  opacity: 0.28,
  depthWrite: false,
  side: THREE.DoubleSide,
});

function buildRootMesh(cone) {
  if (!cone?.vertices?.length || !cone?.faces?.length) return null;
  const g = new THREE.BufferGeometry();
  const pos = new Float32Array(cone.vertices.length * 3);
  cone.vertices.forEach((p, i) => { pos[i * 3] = p[0]; pos[i * 3 + 1] = p[1]; pos[i * 3 + 2] = p[2]; });
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setIndex(cone.faces.flat());
  const m = new THREE.Mesh(g, rootMaterial());
  m.renderOrder = 2;
  m.userData.isVirtualRoot = true;          // never a pick target
  return m;
}

/**
 * Socket caps, flat-shaded.
 *
 * The fan is a triangle soup that meets the arch at the cervical margin — a
 * genuine hard edge. Smooth normals there average the cap into the crown wall
 * and produce the dark blurry smudge instead of a crisp alveolar depression.
 * flatShading derives the normal per triangle in the shader, so the margin
 * stays a clean crease no matter how the fan is tessellated, and the fan's
 * shared centroid vertex cannot smear shading across the whole cap.
 */
const socketMaterial = () => tissueMaterial({
  flatShading: true,
  roughness: 0.62,        // bone/tissue, not enamel — no wet highlight
  clearcoat: 0.0,
});

/**
 * Remove an extracted tooth's faces from the arch and close the socket.
 *
 * The arch's position and colour buffers are never touched — only which faces
 * the index buffer draws. That is what makes extraction cheap (the backend
 * sends face indices, not a re-capped 15MB mesh) and what keeps every
 * client-held vertex id valid across cuts.
 *
 * Socket caps accumulate in a SEPARATE mesh rather than being appended to the
 * arch. Growing the arch's attribute arrays would mean reallocating position,
 * colour and normal on every cut and re-deriving baseColors; a small sibling
 * mesh with the same tissue material is visually identical and leaves the arch
 * buffers pristine. It is kept out of the pick list, so sockets are inert.
 */
function applyExtraction(archRec, removedFaces, socketCap, scene) {
  const geom = archRec.geometry;
  const { fullIndex, liveFaces } = geom.userData;

  for (const fi of removedFaces) {
    if (fi >= 0 && fi < liveFaces.length) liveFaces[fi] = 0;
  }

  let live = 0;
  for (let i = 0; i < liveFaces.length; i++) if (liveFaces[i]) live++;
  const next = new Uint32Array(live * 3);
  let w = 0;
  for (let i = 0; i < liveFaces.length; i++) {
    if (!liveFaces[i]) continue;
    next[w++] = fullIndex[i * 3];
    next[w++] = fullIndex[i * 3 + 1];
    next[w++] = fullIndex[i * 3 + 2];
  }
  geom.setIndex(new THREE.BufferAttribute(next, 1));
  // The index just changed. A stale tree would keep reporting hits on
  // triangles that are no longer drawn — a click landing on a tooth that has
  // already left the cast.
  refreshBoundsTree(geom);
  geom.computeVertexNormals();          // orphaned vertices simply go unused

  // The brush must forget the tooth that just left the cast.
  archRec.brushIndex = new BrushIndex(geom, CELL_FACTOR);
  geom.brushIndex = archRec.brushIndex;

  if (socketCap?.faces?.length) {
    const pos = geom.attributes.position;
    const extra = socketCap.vertices || [];
    // Two index spaces in one face list: a non-negative index is a rim vertex
    // in the arch's own buffer (unchanged, as always), and -(k+1) is the k-th
    // vertex the cup created. Resolving both here keeps the arch's position
    // buffer untouched, which is what every client-held vertex id depends on.
    const xyz = (i) => (i >= 0
      ? [pos.getX(i), pos.getY(i), pos.getZ(i)]
      : extra[-i - 1]);
    for (const tri of socketCap.faces) {
      for (const i of tri) {
        const p = xyz(i);
        archRec.socketVerts.push(p[0], p[1], p[2]);
      }
    }

    disposeMesh(archRec.socketMesh);
    const sg = new THREE.BufferGeometry();
    const arr = new Float32Array(archRec.socketVerts);
    sg.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    const col = new Float32Array(arr.length);
    for (let i = 0; i < arr.length / 3; i++) GINGIVA.toArray(col, i * 3);
    sg.setAttribute("color", new THREE.BufferAttribute(col, 3));

    // No computeVertexNormals(): the material is flatShading, so normals are
    // derived per triangle in the shader. Supplying smoothed vertex normals
    // here would be dead data at best and, on the shared fan centroid, exactly
    // the averaging that smudged the socket.
    archRec.socketMesh = new THREE.Mesh(sg, socketMaterial());
    archRec.socketMesh.userData.isSocketCap = true;   // never a pick target
    scene.add(archRec.socketMesh);
  }
}

const PALETTE = [
  "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
  "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
  "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000"
];

const FDI_COLOURS = {};
[11,12,13,14,15,16,17,18, 21,22,23,24,25,26,27,28, 31,32,33,34,35,36,37,38, 41,42,43,44,45,46,47,48].forEach((fdi, i) => {
    FDI_COLOURS[fdi] = new THREE.Color(PALETTE[i % PALETTE.length]).toArray();
});

/**
 * The six clinical degrees of freedom, with the step each one is prescribed in.
 * Steps are clinical conventions, not UI taste: angulation and inclination are
 * charted to the half degree, rotation to the degree, translations to a tenth
 * of a millimetre.
 */
const CONTROLS = [
  { key: "tip_deg",      label: "Tip / Angulation",     unit: "°",  step: 0.5 },
  { key: "torque_deg",   label: "Torque / Inclination", unit: "°",  step: 0.5 },
  { key: "rotation_deg", label: "Rotation",             unit: "°",  step: 1.0 },
  { key: "d_md",         label: "Mesiodistal",          unit: "mm", step: 0.1 },
  { key: "d_bl",         label: "Buccolingual",         unit: "mm", step: 0.1 },
  { key: "d_oa",         label: "Intrusion / Extrusion", unit: "mm", step: 0.1 },
];

const decimalsFor = (step) => (step >= 1 ? 0 : String(step).split(".")[1].length);

/**
 * What clear aligners typically DELIVER, per channel, for a single course.
 *
 * DISPLAY ONLY. Nothing here gates, clamps or refuses anything — the clinician
 * reads the number and decides, which is the same rule staging_estimate follows
 * in returning counts and no recommendations.
 *
 * Why this exists: a shipped manifest carried d_oa 7.686mm and reported "31
 * stages required" with nothing to say that extrusion of that size is far
 * outside what aligners achieve. The stage count answered "how many trays" and
 * left "is this deliverable at all" unasked.
 *
 * `perStage` comes from the engine — staging_estimate's own defaults, 0.25mm
 * and 2 deg per aligner. `typical` is CLINICAL CONVENTION for a full course and
 * is not something this codebase can measure, so it lives here in one table, is
 * labelled as convention, and is meant to be edited by the practice that owns
 * it rather than treated as ground truth.
 */
const REFERENCE_BANDS = {
  tip_deg:      { typical: [0, 15], note: "crown tipping is among the more predictable channels" },
  torque_deg:   { typical: [0, 12], note: "root torque expresses poorly; plan overcorrection" },
  rotation_deg: { typical: [0, 15], note: "round teeth (canines, premolars) resist rotation" },
  d_md:         { typical: [0, 3],  note: "needs space — check interproximal clearance" },
  d_bl:         { typical: [0, 2],  note: "" },
  d_oa:         { typical: [0, 2],  note: "extrusion is the least predictable movement there is" },
};
const PER_STAGE = { deg: 2.0, mm: 0.25 };   // staging_estimate's own defaults

/**
 * One clinical value: −  [ number ]unit  +
 *
 * The text field is UNCONTROLLED while focused. Round-tripping every keystroke
 * through the transform would rewrite the box mid-edit — typing "-" or "0."
 * parses to nothing and the field would fight the clinician. So local text is
 * kept during editing and committed on blur/Enter, while gizmo drags write
 * straight through whenever the field is not focused.
 */
function ClinicalInput({ spec, value, disabled, onCommit }) {
  const [text, setText] = useState("");
  const [editing, setEditing] = useState(false);
  const shown = editing ? text : value.toFixed(decimalsFor(spec.step));

  const commit = (raw) => {
    const n = Number.parseFloat(raw);
    onCommit(Number.isFinite(n) ? n : value);      // reject junk, keep last good
    setEditing(false);
  };
  const bump = (dir) => {
    const d = decimalsFor(spec.step);
    onCommit(Number((value + dir * spec.step).toFixed(d)));
  };

  return (
    <div style={{ marginBottom: 8 }}>
      <div style={S.dt}>{spec.label}</div>
      <div style={{ display: "flex", gap: 4, alignItems: "stretch" }}>
        <button type="button" disabled={disabled} onClick={() => bump(-1)}
                title={`−${spec.step}${spec.unit}`} style={S.stepBtn}>−</button>
        <div style={S.inputWrap}>
          <input
            type="text" inputMode="decimal" disabled={disabled} value={shown}
            onChange={(e) => { setEditing(true); setText(e.target.value); }}
            onFocus={(e) => { setEditing(true); setText(shown); e.target.select(); }}
            onBlur={(e) => commit(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") { e.currentTarget.blur(); }
              else if (e.key === "Escape") { setEditing(false); e.currentTarget.blur(); }
              else if (e.key === "ArrowUp") { e.preventDefault(); setEditing(false); bump(1); }
              else if (e.key === "ArrowDown") { e.preventDefault(); setEditing(false); bump(-1); }
            }}
            style={S.numInput} />
          <span style={S.unit}>{spec.unit}</span>
        </div>
        <button type="button" disabled={disabled} onClick={() => bump(1)}
                title={`+${spec.step}${spec.unit}`} style={S.stepBtn}>+</button>
      </div>
      <ReferenceBand spec={spec} value={value} />
    </div>
  );
}

/** The typical range beside the requested value. Informational, never a gate. */
function ReferenceBand({ spec, value }) {
  const band = REFERENCE_BANDS[spec.key];
  if (!band) return null;
  const [lo, hi] = band.typical;
  const mag = Math.abs(value);
  const over = mag > hi;
  const stages = Math.ceil(mag / (spec.unit === "mm" ? PER_STAGE.mm : PER_STAGE.deg));
  return (
    <div style={{ fontSize: 10, lineHeight: 1.4, marginTop: 2,
                  color: over ? "#e8a06a" : "#6c7480" }}>
      typical {lo}–{hi}{spec.unit}
      {mag > 0 && ` · requested ${mag.toFixed(decimalsFor(spec.step))}${spec.unit} · ${stages} stage${stages === 1 ? "" : "s"}`}
      {over && ` — ${(mag / hi).toFixed(1)}× the typical ceiling`}
      {band.note && <div style={{ color: "#5a616b" }}>{band.note}</div>}
    </div>
  );
}

export default function App() {
  const mountRef = useRef(null);
  const three = useRef({});
  const arches = useRef({});
  const teeth = useRef({});
  // Per-vertex FDI labels from /segment, kept per arch. They used to be turned
  // into vertex colours and thrown away, which left the client unable to say
  // WHICH tooth a selection covers — and therefore unable to pick the right
  // root length before the cut, since /cut needs it in the request.
  const labels = useRef({});
  const stateRef = useRef();
  const gizmoRef = useRef(null);

  const [status, setStatus] = useState("Load an arch to begin.");
  const [sessions, setSessions] = useState({});
  const [active, setActive] = useState("mandibular");
  const [busy, setBusy] = useState(false);
  // Segmentation is tracked SEPARATELY from `busy`. loadArch, executeCut,
  // exportSetup and exportStages all set `busy`, and the AI status poll writes
  // the status line every second — gating that poll on `busy` meant a machine
  // where the model had failed to load reported "AI unavailable" over the top
  // of a running 40s export. `segmenting` is true only for an actual AI run.
  const [segmenting, setSegmenting] = useState(false);
  // Local wall-clock start of the current AI run, and the poll's kill switch:
  // it is nulled synchronously in runSegmentation's finally, so a fetch still
  // in flight when the run ends cannot overwrite the completion message.
  const segmentStartedAt = useRef(null);
  // Always-on backend health, independent of `busy` and `segmenting`. It writes
  // ONLY its own state and never touches setStatus — writing the status line
  // from a background poll is exactly what made the old one stomp on export
  // messages.
  const [health, setHealth] = useState({ state: "checking", detail: "" });

  // Attachment placement. Kept out of `tool` because it is a MODE over the
  // selected crown rather than another selection brush - the wand and brush
  // paint the arch, this one bonds to a tooth that has already been cut.
  const [attachMode, setAttachMode] = useState(false);
  const [attachSettings, setAttachSettings] = useState({
    shape: "vertical_rectangular",
    dimensions: { ...{ md: ATTACHMENT_SHAPES.vertical_rectangular.md,
                       oa: ATTACHMENT_SHAPES.vertical_rectangular.oa,
                       bl: ATTACHMENT_SHAPES.vertical_rectangular.bl } },
    rotation_deg: 0,
  });
  const [placedAttachments, setPlacedAttachments] = useState([]);
  const attachToolRef = useRef(null);
  // Mirrored for the imperative pointer handlers, which are installed once and
  // must read the CURRENT settings without being re-installed on every slider
  // move. Written in an EFFECT, not during render: React may discard a render,
  // and `activeTooth` is declared further down this component, so reading it
  // here during render would be a temporal-dead-zone crash - the fourth time
  // that trap has appeared in this file.
  const attachRef = useRef({ mode: false, settings: null, tooth: null });
  // The placement callback, reached through a ref for two reasons: the scene
  // effect must run ONCE (depending on a callback would tear down and rebuild
  // three.js on every change), and placeAttachment is declared further down
  // this component, so naming it in a deps array would be a temporal-dead-zone
  // crash - the fifth occurrence of that trap here.
  const placeAttachmentRef = useRef(null);

  const [archFrame, setArchFrame] = useState(null);
  const [tool, setTool] = useState("wand"); 
  const [tolerance, setTolerance] = useState(1.2);
  const [radius, setRadius] = useState(1.5);
  const [selection, setSelection] = useState([]);
  
  const [pickMode, setPickMode] = useState(null); 
  const [mesialPt, setMesialPt] = useState(null);
  const [distalPt, setDistalPt] = useState(null);
  // Fallback only, for when /segment has not run and no FDI can be derived.
  // Per-tooth root length lives on the tooth record — a canine's root is 13mm
  // and a molar's 9mm, and one session-wide slider put a mandibular canine on
  // 10mm in a shipped manifest.
  const [rootLength, setRootLength] = useState(10.0);
  const [activeTooth, setActiveTooth] = useState(null);

  useEffect(() => {
    attachRef.current = { mode: attachMode, settings: attachSettings, tooth: activeTooth };
    attachToolRef.current?.refresh();
    if (!attachMode) attachToolRef.current?.hide();
  }, [attachMode, attachSettings, activeTooth]);

  const [gizmoMode, setGizmoMode] = useState("rotate");
  // Stage scrubbing. `stage` is React state because the chrome shows it, but
  // the POSE is written straight into three.js from a ref (see applyStage) —
  // nothing per-frame goes through the component tree.
  const [stage, setStage] = useState(0);
  const [staging, setStaging] = useState({ total: 0, perTooth: [] });
  const [kinematics, setKinematics] = useState({
    tip_deg: 0, torque_deg: 0, rotation_deg: 0, d_md: 0, d_bl: 0, d_oa: 0
  });

  useEffect(() => {
    stateRef.current = {
      tool, radius, arches: arches.current, activeArch: active, 
      selection, controls: three.current.controls, 
      camera: three.current.camera, raycaster: three.current.raycaster, 
      sessions, activeTooth
    };
  });

  useEffect(() => {
    const mount = mountRef.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d0f12);

    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);

    // WebGLRenderer THROWS if the browser cannot give it a context — a blocked
    // GPU, a software-rendering blacklist, or simply too many live contexts
    // because the tab has been reloaded repeatedly. Unguarded, that throw
    // escapes the effect and the clinician gets the blank page the error
    // boundary exists to prevent, with no statement of the cause.
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch (err) {
      try {
        // Antialiasing needs a multisampled buffer. Dropping it is often the
        // difference between a context and none on constrained hardware, so it
        // is worth one retry before giving up on the viewport entirely.
        renderer = new THREE.WebGLRenderer({ antialias: false });
        console.warn(`[WebGL] antialiasing unavailable (${err.message}); `
                     + `continuing without it.`);
      } catch (err2) {
        setStatus(`3D viewport unavailable: this browser could not create a WebGL `
                  + `context (${err2.message}). The backend is unaffected — an `
                  + `existing case can still be exported.`);
        return () => {};
      }
    }
    renderer.setPixelRatio(window.devicePixelRatio);
    // Physically-based materials need tone mapping to stay off the clipping
    // ceiling; without it the clearcoat highlight blows out to flat white and
    // the enamel reads as plastic.
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // --- dental studio rig -------------------------------------------------
    // Parented to the CAMERA, not the scene. Scanner axes are arbitrary — that
    // is the whole reason arch_frame exists — so a world-fixed key light points
    // somewhere different for every case and leaves half the arches lit from
    // underneath. Camera-relative lighting is what CAD viewers do: the model is
    // always lit from over the operator's shoulder however the case is oriented,
    // and the rig costs nothing to keep aligned.
    const key = new THREE.DirectionalLight(0xfff6ec, 2.1);   // warm key, 45 deg up-left
    key.position.set(-1, 1, 1);
    const fill = new THREE.DirectionalLight(0xdce9ff, 0.55); // cool fill, opposite
    fill.position.set(1.4, -0.6, 0.8);
    const rim = new THREE.DirectionalLight(0xffffff, 0.9);   // rim, behind and above
    rim.position.set(0.2, 1.1, -1.4);
    camera.add(key, fill, rim);
    scene.add(camera);                       // a camera must be in the graph to
                                             // carry children through the frame
    scene.add(new THREE.AmbientLight(0xffffff, 0.18));   // floor, not a fill

    // --- shadow caster, aligned to the PATIENT ----------------------------
    // The rig above is camera-parented on purpose, and a shadow cannot be: a
    // shadow that swings with the camera reads as the model moving, not the
    // light. So the caster is world-fixed — but NOT to world -Y, which is a
    // scanner axis and therefore arbitrary. It is aimed along the arch frame's
    // -u_occ once the occlusal plane is established, which is fixed relative to
    // the patient and independent of how the scan was oriented. Until then it
    // sits on the scene's own PCA-free default and simply does not cast.
    const sun = new THREE.DirectionalLight(0xffffff, 0.0);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    sun.shadow.camera.near = 1;
    sun.shadow.camera.far = 400;
    sun.shadow.bias = -0.0015;
    scene.add(sun, sun.target);

    const catcher = new THREE.Mesh(
      new THREE.PlaneGeometry(400, 400),
      new THREE.ShadowMaterial({ opacity: 0.22 }));
    catcher.receiveShadow = true;
    catcher.visible = false;                 // shown once the plane is known
    catcher.userData.isShadowCatcher = true; // never a pick target
    scene.add(catcher);

    // autoUpdate OFF. A shadow map re-rendered every frame is the single most
    // expensive thing in this scene and nothing in it moves during an orbit —
    // the geometry only changes on a cut or a committed transform, which is
    // when three.current.shadowsDirty is set and one update is taken.
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.shadowMap.autoUpdate = false;

    // Image-based lighting. A physical material with no environment has nothing
    // to reflect, so the clearcoat has no highlight to modulate and the whole
    // cast looks matte and dead however the lights are set.
    const pmrem = new THREE.PMREMGenerator(renderer);
    const envRT = pmrem.fromScene(new RoomEnvironment(), 0.04);
    scene.environment = envRT.texture;
    scene.environmentIntensity = 0.45;       // support, not the main source
    pmrem.dispose();


    installBVH();   // patch three's raycast before anything is picked
    // sun AND catcher GO IN THE LITERAL. They used to be assigned onto
    // three.current a few lines above — and then this statement REPLACED the
    // whole object, so both were gone by the time anything read them.
    // `aimShadows` opens with `if (!sun || !catcher) return;`, so the entire
    // shadow rig silently never armed: no throw, no warning, just no shadow.
    // That is the same class of failure as the record-vs-Object3D mistake below
    // — the silent half of a bug is the half that survives a fix.
    three.current = { scene, camera, renderer, controls, sun, catcher,
                      // firstHitOnly is not a micro-optimisation: without it
                      // three-mesh-bvh collects and sorts EVERY intersection
                      // along the ray, and all five call sites take [0]. This
                      // is the one raycaster the brush, the gizmo, the frame
                      // picker and the attachment tool all share.
                      raycaster: configureRaycaster(new THREE.Raycaster()) };
    const detach = attachResize(mount, camera, renderer);

    // Click-to-place attachments. Installed once; it reads live settings through
    // attachRef so a slider move does not re-install pointer handlers. It raycasts
    // ONLY against cut crowns - an attachment bonds to a tooth, not to the cast.
    attachToolRef.current = installAttachmentTool({
      dom: renderer.domElement, camera, raycaster: three.current.raycaster, scene,
      getTargets: () => (attachRef.current.mode
        ? Object.values(teeth.current).map((r) => r.mesh).filter(Boolean) : []),
      getSettings: () => attachRef.current.settings,
      onPlace: (p) => { if (attachRef.current.mode) placeAttachmentRef.current?.(p); },
    });

    // Initialize the Deltaface-style 3D Gizmo
    gizmoRef.current = new ToothGizmo(
      scene, camera, renderer.domElement, controls,
      (clinicalValues, _matrixRowMajor) => {
        setKinematics(clinicalValues); // Live UI update during drag
      },
      async (clinicalValues, _matrixRowMajor) => {
        setKinematics(clinicalValues);
        const st = stateRef.current;
        const sid = st.sessions[st.activeArch]?.session_id;
        // Cache the committed pose BEFORE anything else can detach the gizmo.
        // This record is what makes the next cut non-destructive.
        const rec = teeth.current[st.activeTooth];
        if (rec) {
          rec.delta.copy(gizmoRef.current.deltaMatrix());
          rec.clinical = clinicalValues;
        }
        // A tooth moved, so its shadow is wrong until the map is re-rendered.
        three.current.shadowsDirty = true;
        if (sid && st.activeTooth) {
          await fetch(`${API}/api/session/${sid}/tooth/${st.activeTooth}/kinematics`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(clinicalValues)
          }).catch(console.error);
        }
      }
    );

    const cleanupBrush = installBrush(
      mount,
      () => stateRef.current,
      (newSet) => {
        setSelection([...newSet]);
        highlightSelection(stateRef.current.arches[stateRef.current.activeArch].geometry, [...newSet]);
      },
      (finalArray) => {
        const sid = stateRef.current.sessions[stateRef.current.activeArch]?.session_id;
        if(sid) {
          fetch(`${API}/api/session/${sid}/selection`, {
            method:"PUT", headers:{"Content-Type":"application/json"},
            body: JSON.stringify({vertex_ids: finalArray})
          });
        }
      }
    );

    // --- WebGL context loss ------------------------------------------------
    // A context is lost on a GPU driver reset, a laptop switching graphics
    // cards, or the browser reclaiming one from a backgrounded tab. Every GPU
    // resource — geometries, textures, the shadow map, the bounds trees' host
    // buffers — is invalidated, and rendering afterwards throws once per frame.
    //
    // The DEFAULT browser behaviour is the problem: without preventDefault the
    // context is never restored and the canvas stays black forever. So this
    // takes the event, stops the render loop, and says plainly that the CASE IS
    // INTACT — the scan, the cuts and the prescriptions live in the session on
    // the backend, and none of them are GPU state.
    const canvas = renderer.domElement;
    const onContextLost = (e) => {
      e.preventDefault();                 // opt in to restoration
      three.current.contextLost = true;
      console.warn("[WebGL] context lost — suspending the render loop until restore.");
      setStatus("3D context lost (GPU driver reset or the tab was reclaimed). "
                + "Your case is safe on the backend. Waiting for the browser to "
                + "restore the context; reload if it does not return.");
    };
    const onContextRestored = () => {
      three.current.contextLost = false;
      // Every cached shadow map died with the context, and autoUpdate is off,
      // so without this the scene comes back lit but with no shadows at all.
      three.current.shadowsDirty = true;
      console.warn("[WebGL] context restored — resuming.");
      setStatus("3D context restored.");
    };
    canvas.addEventListener("webglcontextlost", onContextLost, false);
    canvas.addEventListener("webglcontextrestored", onContextRestored, false);

    // Shadow allocation is the one thing here that can fail on an otherwise
    // working context: the 1024x1024 depth target is a real buffer and a
    // constrained GPU can refuse it. It is allocated on the FIRST RENDER, not
    // when shadowMap.enabled is set, so this is the only place that can catch
    // it. Shadows are presentation — losing them costs a depth cue, not the
    // viewport — so the failure disables them once and says so, rather than
    // throwing sixty times a second into a console nobody has open.
    let renderFailures = 0;

    let raf;
    (function animate() {
      raf = requestAnimationFrame(animate);
      if (three.current.contextLost) return;
      controls.update();
      // One shadow-map update per change, not per frame. shadowsDirty is set by
      // a cut, a committed transform (dragged or typed), an attachment being
      // bonded or cleared, and the occlusal plane being established — every
      // path that changes geometry. This comment used to claim all of that
      // while aimShadows was the ONLY writer, so every cut and every movement
      // left the pre-cut arch's shadow on the catcher. An orbit moves the
      // camera and nothing else, so it still costs nothing.
      if (three.current.shadowsDirty) {
        renderer.shadowMap.needsUpdate = true;
        three.current.shadowsDirty = false;
      }
      try {
        renderer.render(scene, camera);
        renderFailures = 0;
      } catch (err) {
        renderFailures += 1;
        if (renderer.shadowMap.enabled) {
          renderer.shadowMap.enabled = false;
          sun.castShadow = false;
          catcher.receiveShadow = false;
          three.current.shadowsUnavailable = true;
          console.warn(`[WebGL] render failed (${err.message}); disabling shadows `
            + `and retrying. Geometry, cutting, staging and export are unaffected.`);
        } else if (renderFailures === 1 || renderFailures === 30) {
          // Not the shadow map, then. Report it twice — once immediately and
          // once after half a second of failures — and keep the loop alive, so
          // the sidebar and the export button still work while the viewport
          // does not.
          console.error("[WebGL] render failed with shadows already off:", err);
          setStatus(`3D rendering failed: ${err.message}. The case is intact on the `
            + `backend and export still works; reload to rebuild the viewport.`);
        }
      }
    })();

    return () => {
      cancelAnimationFrame(raf);
      canvas.removeEventListener("webglcontextlost", onContextLost);
      canvas.removeEventListener("webglcontextrestored", onContextRestored);
      attachToolRef.current?.dispose();
      detach();
      cleanupBrush();
      gizmoRef.current?.detach();
      // Free every buffer this component put on the GPU. Three keeps geometries
      // and materials alive until they are explicitly disposed, so unmounting
      // without this leaks the whole case.
      for (const rec of Object.values(teeth.current)) { disposeMesh(rec.mesh); disposeMesh(rec.rootMesh); }
      teeth.current = {};
      for (const a of Object.values(arches.current)) {
        disposeMesh(a.mesh); disposeMesh(a.socketMesh);
      }
      arches.current = {};
      scene.environment?.dispose();
      controls.dispose(); renderer.dispose();
      if (renderer.domElement.parentNode) mount.removeChild(renderer.domElement);
    };
  }, []);

  const highlightSelection = (geom, vertexIds) => {
    if (!geom.userData.baseColors) return;
    const attr = geom.attributes.color;
    attr.array.set(geom.userData.baseColors); 
    for (const vId of vertexIds) SELECTED.toArray(attr.array, vId * 3);
    attr.needsUpdate = true;
  };

  /**
   * The case validation rows, derived from live state.
   *
   * Every row must be able to say "not determined". The temptation is to show a
   * tick when nothing is wrong, but "nothing is wrong" and "nothing was checked"
   * are different findings and only one of them is reassuring. The antagonist
   * row is the sharp case: it is UNKNOWN until an opposing arch is loaded, and
   * a green tick there would claim the bite was verified.
   */
  const caseChecks = useMemo(() => {
    const rows = [];
    const archNames = Object.keys(sessions);
    const toothList = Object.values(teeth.current);

    rows.push(archNames.length
      ? { id: "scans", state: PASS, label: `${archNames.length} arch scan(s) loaded`,
          detail: archNames.join(", ") }
      : { id: "scans", state: UNKNOWN, label: "No arch loaded" });

    const health = sessions[active]?.scan_health;
    if (health) {
      const nm = health.nonmanifold_edges ?? 0;
      rows.push({
        id: "mesh",
        state: nm > 0 ? REVIEW : PASS,
        label: nm > 0 ? "Scan has non-manifold edges" : "Mesh integrity",
        detail: `${health.open_edges ?? 0} open, ${nm} non-manifold. An intraoral scan is an `
              + `open shell by nature — open edges are its perimeter, not damage.`,
      });
    } else {
      rows.push({ id: "mesh", state: UNKNOWN, label: "Mesh integrity not measured" });
    }

    rows.push(archFrame
      ? { id: "reference", state: PASS, label: "Occlusal reference established" }
      : { id: "reference", state: UNKNOWN,
          label: "Occlusal reference not set",
          detail: "Required before any cut — C_res is extrapolated along an axis "
                + "reconciled against it." });

    rows.push(labels.current[active]
      ? { id: "segmentation", state: REVIEW, label: "Segmentation present — review advised",
          detail: "AI tooth identification has no validated accuracy figure in this build. "
                + "Confirm each FDI before relying on a per-tooth root length.",
          basis: "software heuristic" }
      : { id: "segmentation", state: UNKNOWN, label: "Segmentation not run",
          detail: "FDI is unknown, so root length falls back to the session slider." });

    if (!toothList.length) {
      rows.push({ id: "teeth", state: UNKNOWN, label: "No teeth extracted yet" });
    } else {
      const unnamed = toothList.filter((t) => t.fdi == null).length;
      rows.push({
        id: "teeth",
        state: unnamed ? REVIEW : PASS,
        label: unnamed ? `${unnamed} of ${toothList.length} teeth unidentified`
                       : `${toothList.length} tooth/teeth identified`,
        detail: unnamed ? "An unidentified tooth used the session default root length."
                        : undefined,
      });
      rows.push({ id: "cres", state: PASS, label: "C_res defined for every extracted tooth",
                  detail: "Each pivot was checked to sit inside the alveolus at cut time." });
    }

    // Antagonist: UNKNOWN unless an opposing arch exists AND a check has run.
    const opposing = active === "maxillary" ? "mandibular" : "maxillary";
    const measured = toothList.filter((t) => t.occlusion);
    if (!sessions[opposing]) {
      rows.push({ id: "antagonist", state: UNKNOWN, label: "Antagonist check not available",
                  detail: `No ${opposing} arch is loaded. This is NOT a finding of no interference.` });
    } else if (!measured.length) {
      rows.push({ id: "antagonist", state: UNKNOWN, label: "Antagonist not yet checked",
                  detail: "Commit a movement to measure clearance against the opposing arch." });
    } else {
      const worst = measured.reduce(
        (m, t) => Math.max(m, t.occlusion?.max_penetration_mm ?? 0), 0);
      rows.push({
        id: "antagonist",
        state: worst > 0.1 ? REVIEW : PASS,
        label: worst > 0.1 ? `Occlusal interference ${worst.toFixed(2)}mm`
                           : "No antagonist interference detected",
        detail: "Nearest-vertex signed distance — approximate, and it cannot see an "
              + "intersection between sampled vertices.",
        basis: "software heuristic",
      });
    }

    if (staging.total > 0) {
      rows.push({ id: "staging", state: PASS,
                  label: `${staging.total} stages planned`,
                  detail: "Each stage is the prescription scaled and rebuilt, never an "
                        + "interpolated matrix." });
    }
    return rows;
    // activeTooth and kinematics look unnecessary to the linter and are not.
    // This reads teeth.current, a REF — React cannot see it mutate, so cutting
    // or moving a tooth would leave the panel stale forever. These two are the
    // change signals for exactly those events. eslint cannot know that a ref
    // read needs an external trigger, so the rule is disabled here rather than
    // the dependency removed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessions, active, archFrame, staging, activeTooth, kinematics]);

  /** Send a clicked placement to the backend and fuse it onto the crown. */
  const placeAttachment = useCallback(async (payload) => {
    const st = attachRef.current;
    const sid = stateRef.current?.sessions?.[stateRef.current.activeArch]?.session_id;
    const tid = payload.tooth_id || st.tooth;
    if (!sid || !tid) { setStatus("Select a cut tooth before placing an attachment."); return; }
    try {
      const res = await fetch(`${API}/api/session/${sid}/tooth/${tid}/attachment`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...payload, tooth_id: tid }),
      });
      const body = await res.text();
      if (!res.ok) {
        // 422 carries the reason - implausible size, or not touching the crown.
        let msg = body;
        try { msg = JSON.parse(body).detail || body; } catch { /* plain text */ }
        setStatus(`Attachment refused: ${msg}`);
        return;
      }
      const data = JSON.parse(body);
      setPlacedAttachments(data.attachments || []);

      // Replace the crown geometry with the FUSED solid, so what the clinician
      // sees is the solid that will be staged and printed.
      const rec = teeth.current[tid];
      if (rec?.mesh && data.crown) {
        const g = rec.mesh.geometry;
        g.setAttribute("position",
          new THREE.BufferAttribute(new Float32Array(data.crown.positions), 3));
        g.setIndex(new THREE.BufferAttribute(new Uint32Array(data.crown.indices), 1));
        g.computeVertexNormals();
        const n = g.attributes.position.count;
        const c = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) CROWN_COL.toArray(c, i * 3);
        g.setAttribute("color", new THREE.BufferAttribute(c, 3));
        // allowWeld: a crown's vertex ids are local. It is rebuilt wholesale
        // from each server payload and nothing keys a selection on it, unlike
        // the arch, where welding would re-point the wand at other anatomy.
        refreshBoundsTree(g, { allowWeld: true });
        // The crown is a different solid now - it has a bump on it. Same
        // reason as a cut: autoUpdate is off, so without this the catcher
        // keeps the silhouette of the un-bonded crown.
        three.current.shadowsDirty = true;
      }
      setStatus(`${data.attachment.shape.replace(/_/g, " ")} bonded — `
                + `fused to ${data.bodies} body, ${data.volume_mm3}mm3.`);
    } catch (err) {
      setStatus(`Attachment failed: ${err.message}`);
    }
  }, []);

  // Published in an effect BELOW its own declaration. Putting it in the deps
  // array of the effect further up crashed the app with "Cannot access
  // 'placeAttachment' before initialization" - deps arrays are evaluated during
  // render, while a const declared later is still in its temporal dead zone.
  // npm run smoke caught it; npm run build did not, and reported success.
  useEffect(() => { placeAttachmentRef.current = placeAttachment; }, [placeAttachment]);

  const clearAttachments = useCallback(async () => {
    const sid = stateRef.current?.sessions?.[stateRef.current.activeArch]?.session_id;
    const tid = attachRef.current.tooth;
    if (!sid || !tid) return;
    const res = await fetch(`${API}/api/session/${sid}/tooth/${tid}/attachment`,
                            { method: "DELETE" });
    if (!res.ok) return;
    const data = await res.json();
    setPlacedAttachments([]);
    const rec = teeth.current[tid];
    if (rec?.mesh && data.crown) {
      const g = rec.mesh.geometry;
      g.setAttribute("position",
        new THREE.BufferAttribute(new Float32Array(data.crown.positions), 3));
      g.setIndex(new THREE.BufferAttribute(new Uint32Array(data.crown.indices), 1));
      g.computeVertexNormals();
      refreshBoundsTree(g, { allowWeld: true });
      // Removing the bumps changes the silhouette back. The shadow map has to
      // be told; nothing else in the frame loop notices a geometry swap.
      three.current.shadowsDirty = true;
    }
    setStatus(`Cleared ${data.cleared} attachment(s); crown restored as cut.`);
  }, []);

  /**
   * Case stage count = MAX over committed teeth, plus who binds each one.
   *
   * MUST STAY ABOVE its callers — the restore effect and applyClinical both
   * list it in a dependency array, and dependency arrays are evaluated DURING
   * RENDER, while a `const` declared further down the component is still in its
   * temporal dead zone. Declaring it below produced exactly the white-screen
   * `ReferenceError: Cannot access 'X' before initialization` that
   * opposingSessionId caused. This is the THIRD value to hit that trap; if you
   * add a callback to a deps array, check where it is declared first.
   */
  const refreshStaging = useCallback(() => {
    const rows = Object.entries(teeth.current).map(([tid, rec]) => {
      const s = stagingFor(rec.clinical);
      return { tid, fdi: rec.fdi, stages: s.stages, driver: s.driver, channel: s.channel,
               occlusion: rec.occlusion || null };
    }).filter((r) => r.stages > 0);
    const total = rows.reduce((m, r) => Math.max(m, r.stages), 0);
    rows.forEach((r) => { r.binds = r.stages === total; });
    rows.sort((a, b) => b.stages - a.stages);
    setStaging({ total, perTooth: rows });
    return total;
  }, []);

  /**
   * Remember which session belongs to which arch, across a refresh.
   *
   * This is the ONE thing the server genuinely cannot recover: everything else
   * about a case is in the session, but the session id itself lived only in
   * React state, so a refresh made all of it unreachable. Only the ids are
   * stored — no geometry, no labels, nothing patient-derived leaves memory.
   */
  const rememberSessions = useCallback((next) => {
    try {
      const ids = Object.fromEntries(
        Object.entries(next).filter(([, s]) => s?.session_id)
                            .map(([arch, s]) => [arch, s.session_id]));
      window.localStorage.setItem(CASE_KEY, JSON.stringify(ids));
    } catch { /* private mode / storage disabled — restore is a convenience */ }
    return next;
  }, []);

  /**
   * Build the arch into the scene from a /mesh payload.
   *
   * Extracted so that uploading a scan and RESTORING one after a refresh walk
   * the identical path. Two code paths producing "the same" arch is how a
   * restored case quietly differs from the one that was planned.
   */
  const mountArch = useCallback((archName, m) => {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(m.positions), 3));
    const fullIndex = new Uint32Array(m.indices);
    geom.setIndex(new THREE.BufferAttribute(fullIndex, 1));
    geom.computeVertexNormals();

    const nVerts = geom.attributes.position.count;
    const colors = new Float32Array(nVerts * 3);
    for (let i = 0; i < nVerts; i++) GINGIVA.toArray(colors, i * 3);
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geom.userData.baseColors = Float32Array.from(colors);

    // The pristine index buffer plus a per-face liveness flag. Extraction
    // rewrites the INDEX only — positions, colours and therefore every
    // vertex id the backend and the client exchange stay exactly as loaded.
    geom.userData.fullIndex = fullIndex;
    geom.userData.liveFaces = new Uint8Array(fullIndex.length / 3).fill(1);

    geom.brushIndex = new BrushIndex(geom, CELL_FACTOR);
    // 12.97ms -> 0.023ms per raycast on a 204,800-face mesh (see bvh.js).
    refreshBoundsTree(geom);

    const { scene, camera, controls, renderer } = three.current;
    const prev = arches.current[archName];
    if (prev) {
      disposeMesh(prev.mesh);
      disposeMesh(prev.socketMesh);
      for (const [tid, rec] of Object.entries(teeth.current)) {
        if (rec.archName === archName) { disposeMesh(rec.mesh); disposeMesh(rec.rootMesh); delete teeth.current[tid]; }
      }
    }

    const mesh = new THREE.Mesh(geom, tissueMaterial());
    scene.add(mesh);

    const view = frameArch(geom, camera, controls, renderer);
    arches.current[archName] = { mesh, geometry: geom, view, brushIndex: geom.brushIndex,
                                 socketMesh: null, socketVerts: [], socketFaces: [] };
    return arches.current[archName];
  }, []);

  const loadArch = useCallback(async (event, archName) => {
    const fileObj = event.target.files?.[0];
    if (!fileObj) return;
    setBusy(true); setStatus(`Uploading ${fileObj.name}...`);

    try {
      const fd = new FormData();
      fd.append("arch", archName === "maxillary" ? "upper" : "lower");
      fd.append("file", fileObj);
      
      const resSession = await fetch(`${API}/api/session`, {method:"POST", body:fd});
      if (!resSession.ok) throw new Error(await resSession.text());
      const s = await resSession.json();

      const resMesh = await fetch(`${API}/api/session/${s.session_id}/mesh`);
      const m = await resMesh.json();

      mountArch(archName, m);

      setSessions((prev) => rememberSessions({ ...prev, [archName]: s }));
      setActive(archName);
      setArchFrame(null);
      setStatus(`Arch Loaded. Ready for AI Segmentation or Occlusal Plane definition.`);
    } catch (err) {
      setStatus(`Failed: ${err.message}`);
    } finally { setBusy(false); }
  }, [mountArch, rememberSessions]);

  /**
   * Rebuild one arch, and everything done to it, from its session id alone.
   *
   * The server held all of this the whole time; until the hydration endpoints
   * existed none of it was reachable, so a refresh lost the case. Nothing here
   * recomputes geometry — every crown, socket and pose is the one that was
   * approved, fetched back verbatim.
   */
  const restoreArch = useCallback(async (archName, sid) => {
    const summary = await fetch(`${API}/api/session/${sid}`);
    if (!summary.ok) return null;           // expired or server restarted
    const info = await summary.json();

    const m = await (await fetch(`${API}/api/session/${sid}/mesh`)).json();
    const archRec = mountArch(archName, m);

    if (info.has_segmentation) {
      const r = await fetch(`${API}/api/session/${sid}/labels`);
      if (r.ok) labels.current[archName] = (await r.json()).labels;
    }

    let frame = null;
    if (info.has_occlusal_frame) {
      const r = await fetch(`${API}/api/session/${sid}/frame`);
      if (r.ok) frame = await r.json();
    }

    if (info.tooth_count > 0) {
      const r = await fetch(`${API}/api/session/${sid}/teeth?geometry=true`);
      if (r.ok) {
        const { teeth: rows } = await r.json();
        for (const t of rows) {
          const g = new THREE.BufferGeometry();
          g.setAttribute("position",
            new THREE.BufferAttribute(new Float32Array(t.crown.positions), 3));
          g.setIndex(new THREE.BufferAttribute(new Uint32Array(t.crown.indices), 1));
          refreshBoundsTree(g, { allowWeld: true });
          g.computeVertexNormals();
          const n = g.attributes.position.count;
          const c = new Float32Array(n * 3);
          for (let i = 0; i < n; i++) CROWN_COL.toArray(c, i * 3);
          g.setAttribute("color", new THREE.BufferAttribute(c, 3));

          const mesh = new THREE.Mesh(g, tissueMaterial());
          mesh.userData.toothId = t.tooth_id;
          mesh.matrixAutoUpdate = false;
          three.current.scene.add(mesh);

          const rootMesh = buildRootMesh(t.root_cone);
          if (rootMesh) { rootMesh.matrixAutoUpdate = false; three.current.scene.add(rootMesh); }

          // Re-apply the committed pose. The matrix is rebuilt from the six
          // clinical values rather than trusted as 16 floats, so a restored
          // tooth sits exactly where deltaFromClinical would put it — the same
          // path every staging frame uses.
          const delta = new THREE.Matrix4();
          if (t.clinical) {
            const axes = toothAxes(t.frame);
            delta.copy(deltaFromClinical(axes, t.c_res, t.clinical));
          }
          mesh.matrix.copy(delta); mesh.matrixWorld.copy(delta);
          if (rootMesh) { rootMesh.matrix.copy(delta); rootMesh.matrixWorld.copy(delta); }

          teeth.current[t.tooth_id] = {
            mesh, rootMesh, archName, frame: t.frame, cRes: t.c_res, delta,
            clinical: t.clinical || {tip_deg:0, torque_deg:0, rotation_deg:0, d_md:0, d_bl:0, d_oa:0},
            fdi: t.fdi ?? null,
            rootLength: t.root_length_mm,
            rootDefault: t.fdi != null ? rootDefaultForFDI(t.fdi) : null,
            rootClamp: t.root_cone?.clamped ? t.root_cone : null,
          };

          applyExtraction(archRec, t.removed_faces, t.socket_cap, three.current.scene);
        }
      }
    }
    return { info, frame, session: { session_id: sid, ...info } };
  }, [mountArch]);

  // Restore on mount. Runs once; a failed or expired session is dropped
  // silently rather than leaving a dead id to fail every later request.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      let saved;
      try { saved = JSON.parse(window.localStorage.getItem(CASE_KEY) || "{}"); }
      catch { return; }
      const entries = Object.entries(saved).filter(([, sid]) => typeof sid === "string");
      if (!entries.length) return;

      setBusy(true);
      setStatus("Restoring the previous case...");
      const restored = {};
      let frameSeen = null, teethSeen = 0;
      for (const [archName, sid] of entries) {
        try {
          const out = await restoreArch(archName, sid);
          if (cancelled) return;
          if (out) {
            restored[archName] = out.session;
            if (out.frame) frameSeen = out.frame;
            teethSeen += out.info.tooth_count;
          }
        } catch { /* one bad arch must not abort the other */ }
      }
      if (cancelled) return;
      setBusy(false);

      const names = Object.keys(restored);
      if (!names.length) {
        try { window.localStorage.removeItem(CASE_KEY); } catch { /* ignore */ }
        setStatus("Previous case has expired. Load an arch to begin.");
        return;
      }
      setSessions(rememberSessions(restored));
      setActive(names.includes("mandibular") ? "mandibular" : names[0]);
      if (frameSeen) setArchFrame(frameSeen);
      setStage(refreshStaging());
      setStatus(`Case restored — ${names.length} arch(es), ${teethSeen} tooth/teeth`
                + `${frameSeen ? ", occlusal plane" : ""}. Nothing was recomputed.`);
    })();
    return () => { cancelled = true; };
  }, [restoreArch, rememberSessions, refreshStaging]);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await fetch(`${API}/api/ai/status`);
        if (cancelled) return;
        if (!res.ok) {
          setHealth({ state: "degraded", detail: `API returned ${res.status}` });
          return;
        }
        const s = await res.json();
        if (cancelled) return;
        if (s.warming) {
          setHealth({ state: "warming", detail: "first load takes ~10-40s" });
        } else if (s.loaded) {
          setHealth({ state: "connected", detail: s.warm_seconds ? `AI ready in ${s.warm_seconds}s` : "AI ready" });
        } else {
          // A dead model is NOT a dead backend: upload, wand and manual cutting
          // all still work, so this must not read as a total failure.
          setHealth({ state: "degraded", detail: s.error ? String(s.error).slice(0, 60) : "manual cutting still works" });
        }
      } catch {
        // fetch throws on connection refusal — the API is not listening at all.
        if (!cancelled) setHealth({ state: "offline", detail: "run start_backend.bat" });
      }
    };
    check();
    const timer = setInterval(check, 3000);
    return () => { cancelled = true; clearInterval(timer); };
  }, []);

  // Progress ticker for the AI run. Inference is one long POST with nothing to
  // report until it returns, so the only honest signal is elapsed time plus
  // whatever phase the server says it is in.
  useEffect(() => {
    if (!segmenting) return;
    let cancelled = false;

    const tick = (head, tail) => {
      const t0 = segmentStartedAt.current;
      if (t0 == null) return;   // the run already finished; do not clobber its message
      // Counted from the LOCAL clock, not from the server's
      // `segmentation_started_at`: that is server epoch time, and a browser
      // with a skewed clock would render a negative or absurd number. Local
      // elapsed is also the number the user is actually waiting on.
      setStatus(`${head} — ${Math.max(0, Math.round((Date.now() - t0) / 1000))}s elapsed.${tail}`);
    };

    const poll = async () => {
      if (segmentStartedAt.current == null) return;
      try {
        const res = await fetch(`${API}/api/ai/status`);
        if (cancelled || !res.ok) {
          tick("AI segmenting teeth", " A full arch takes a few minutes.");
          return;
        }
        const s = await res.json();
        if (cancelled) return;
        if (s.warming) {
          tick("Loading the AI model", " First run imports torch and a 64MB checkpoint.");
        } else if (s.segmentation_in_progress) {
          tick(s.segmentation_message || "AI segmenting teeth", " A full arch takes a few minutes.");
        } else if (s.error) {
          // Only reachable while segmenting now, so an export can never be
          // interrupted by an AI problem it does not depend on.
          setStatus(`AI unavailable: ${s.error}`);
        } else {
          tick("AI segmenting teeth", " A full arch takes a few minutes.");
        }
      } catch {
        // --reload may be restarting the backend mid-run. Keep counting rather
        // than flashing an error; runSegmentation's own catch is what reports a
        // genuinely dead backend, once, with the command to fix it.
        if (!cancelled) tick("AI segmenting teeth (waiting on the backend)", "");
      }
    };

    poll();
    const timer = setInterval(poll, 1000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [segmenting]);

  const runSegmentation = useCallback(async () => {
    const sid = sessions[active]?.session_id;
    if (!sid) return;
    if (segmenting) return;   // the server 409s a duplicate run; do not even ask
    setBusy(true);
    setSegmenting(true);
    segmentStartedAt.current = Date.now();
    setStatus("AI segmenting teeth — 0s elapsed. A full arch takes a few minutes.");
    try {
      const res = await fetch(`${API}/api/session/${sid}/segment`, { method: "POST" });
      const bodyText = await res.text();
      if (res.status === 409) {
        // Another run is already in flight on this backend. Not a failure.
        throw new Error("A segmentation run is already in progress on the backend. Wait for it to finish.");
      }
      if (!res.ok) {
        throw new Error(bodyText || "Segmentation request failed.");
      }
      const data = JSON.parse(bodyText);
      const n = arches.current[active].geometry.attributes.position.count;
      const c = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        const [r,g,b] = FDI_COLOURS[data.labels[i]] ?? [0.85,0.75,0.72]; 
        c[i*3]=r; c[i*3+1]=g; c[i*3+2]=b;
      }
      arches.current[active].geometry.setAttribute("color", new THREE.BufferAttribute(c, 3));
      arches.current[active].geometry.userData.baseColors = Float32Array.from(c);
      labels.current[active] = data.labels;
      const took = segmentStartedAt.current
        ? ` (${Math.round((Date.now() - segmentStartedAt.current) / 1000)}s)`
        : "";
      // Stop the ticker BEFORE the completion message, so a poll that is still
      // in flight sees a null start and returns instead of overwriting it.
      segmentStartedAt.current = null;
      setStatus(`Segmentation complete${took} — root lengths will now default per tooth.`);
    } catch (err) {
      const msg = err && err.message ? err.message : String(err);
      segmentStartedAt.current = null;
      if (msg.includes("Failed to fetch") || msg.includes("fetch")) {
        setStatus("Backend unavailable: start the FastAPI API on http://127.0.0.1:8000 (run start_backend.bat)");
      } else {
        setStatus(`Segmentation failed: ${msg}`);
      }
    } finally {
      segmentStartedAt.current = null;
      setSegmenting(false);
      setBusy(false);
    }
  }, [sessions, active, segmenting]);

  const handleDefineOcclusalPlane = async () => {
    const sid = sessions[active]?.session_id;
    if (!sid) return;
    setStatus("Click: 1. Left Molar Cusp, 2. Right Molar Cusp, 3. Anterior Midline");
    try {
      const pts = await pickOcclusalPlane(
        mountRef.current, three.current.camera, three.current.raycaster, arches.current,
        (count, target) => { if (target) setStatus(`Click ${target} (${count}/3)...`); }
      );
      const res = await fetch(`${API}/api/session/${sid}/occlusal-plane`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points: pts })
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setArchFrame(data);
      // Now the patient's own "down" is known, so the shadow can be aimed and
      // the catcher laid under the cast. Before this the caster is dark, because
      // a shadow thrown along an arbitrary scanner axis is worse than none.
      aimShadows(data);
      setStatus(data.midline_warning ? `Warning: ${data.midline_warning}` : "Occlusal Plane Established!");
    } catch (err) {
      setStatus(`Occlusal Plane cancelled: ${err.message}`);
    }
  };

  /**
   * The FDI number this selection sits on, or null.
   *
   * Modal label over the selected vertices. Returns null when /segment has not
   * run — NEVER a guess: _tooth_fdi applies the same rule server-side, because
   * a wrong tooth number on a lab manifest is worse than an absent one, and a
   * root length derived from a wrong number moves C_res into the wrong bone.
   */
  const fdiForSelection = (ids) => {
    const lab = labels.current[active];
    if (!lab || !ids.length) return null;
    const tally = new Map();
    for (const i of ids) {
      const l = lab[i];
      if (l > 0) tally.set(l, (tally.get(l) || 0) + 1);
    }
    if (!tally.size) return null;
    return [...tally.entries()].sort((a, b) => b[1] - a[1])[0][0];
  };

  const executeCut = async () => {
    const sid = sessions[active]?.session_id;
    if (!sid || !selection.length || !mesialPt || !distalPt) return;

    // Root length per TOOTH, derived from its FDI number when segmentation has
    // run. The session-wide slider is the fallback for an unsegmented case.
    const fdi = fdiForSelection(selection);
    const rootDefault = fdi == null ? null : rootDefaultForFDI(fdi);
    const rootUsed = rootDefault ?? rootLength;

    setBusy(true);
    setStatus(fdi == null
      ? `Cutting tooth on a ${rootUsed.toFixed(1)}mm root (no segmentation — session default)...`
      : `Cutting FDI ${fdi} on a ${rootUsed.toFixed(1)}mm root (${fdiClass(fdi)} default)...`);
    try {
      const res = await fetch(`${API}/api/session/${sid}/cut`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({
          vertex_ids: selection, mesial_pt: mesialPt, distal_pt: distalPt,
          root_length_mm: rootUsed
        })
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => null);
        throw new Error(errJson?.detail || await res.text());
      }
      const data = await res.json();

      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(data.crown.positions), 3));
      geom.setIndex(new THREE.BufferAttribute(new Uint32Array(data.crown.indices), 1));
      refreshBoundsTree(geom, { allowWeld: true });
      geom.computeVertexNormals();

      const nVerts = geom.attributes.position.count;
      const colors = new Float32Array(nVerts * 3);
      for (let i = 0; i < nVerts; i++) CROWN_COL.toArray(colors, i * 3);
      geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));

      const mesh = new THREE.Mesh(geom, tissueMaterial());
      mesh.userData.toothId = data.tooth_id;
      three.current.scene.add(mesh);

      // A record, not a bare mesh: the committed delta is what survives the
      // next cut and lets the gizmo resume this tooth where it was left.
      // The fabricated root, drawn beside the crown and driven by the same
      // matrix so it follows every movement and every stage.
      const rootMesh = buildRootMesh(data.root_cone);
      if (rootMesh) three.current.scene.add(rootMesh);

      teeth.current[data.tooth_id] = {
        mesh, rootMesh, archName: active, frame: data.frame, cRes: data.c_res,
        delta: new THREE.Matrix4(),
        clinical: {tip_deg: 0, torque_deg: 0, rotation_deg: 0, d_md: 0, d_bl: 0, d_oa: 0},
        // The server's FDI is authoritative — it reads the same labels through
        // the crown's own face mask rather than through the raw selection.
        fdi: data.fdi ?? fdi,
        rootLength: rootUsed,
        rootDefault,
        // Set only when the DRAWN cone was shortened to stay inside the scan.
        // C_res still uses the full length; this is a note about a picture.
        rootClamp: data.root_cone?.clamped ? data.root_cone : null,
      };

      // TRUE extraction: the tooth's faces leave the cast and the socket is
      // capped in gingiva. Previously this only recoloured the vertices dark,
      // which left the geometry in place and read as an impression.
      // The cast just changed shape. Without this the shadow map keeps the
      // silhouette of the PRE-CUT arch: autoUpdate is off (a shadow re-rendered
      // every frame is the single most expensive thing here), so the map only
      // refreshes when something sets this flag — and until now only
      // aimShadows did, which runs once when the occlusal plane is established.
      three.current.shadowsDirty = true;
      applyExtraction(arches.current[active], data.removed_faces, data.socket_cap,
                      three.current.scene);
      setSelection([]);

      setActiveTooth(data.tooth_id);
      gizmoRef.current.attach(mesh, data.frame, data.c_res, null);
      setKinematics({tip_deg: 0, torque_deg: 0, rotation_deg: 0, d_md: 0, d_bl: 0, d_oa: 0});
      setStage(refreshStaging());

      // Say when the software overrode the tooth's own geometry. A silent
      // correction is how a wrong pivot goes unnoticed until the aligner
      // does not fit.
      const dev = Number.isFinite(data.axis_deviation_deg)
        ? `${data.axis_deviation_deg.toFixed(1)}° off apical` : "deviation unmeasured";
      const height = data.dimensions?.height_oa;
      const axisNote = data.axis_corrected
        ? ` Long axis was ${dev} — corrected (${data.axis_source}). Verify the pivot before staging.`
        : ` Long axis ${dev} (${data.axis_source}).`;
      setStatus(`Tooth cut, C_res anchored in bone.${axisNote}`
                + (Number.isFinite(height) ? ` Crown height ${height.toFixed(1)}mm.` : ""));
    } catch (err) {
      setStatus(`Cut failed: ${err.message}`);
    } finally { setBusy(false); }
  };

  /** Re-attach the gizmo to an already-cut tooth, at the pose it was left in. */
  const selectTooth = (tid) => {
    const rec = teeth.current[tid];
    if (!rec) return;
    setActive(rec.archName);
    setActiveTooth(tid);
    // Passing the committed delta is what makes this resume rather than reset:
    // the pivot starts at delta * rest, so the sliders read cumulative values
    // and a further drag composes on top.
    gizmoRef.current.attach(rec.mesh, rec.frame, rec.cRes, rec.delta);
    setKinematics(rec.clinical);
    const who = rec.fdi != null ? `FDI ${rec.fdi}` : `Tooth ${tid}`;
    const root = rec.rootLength != null
      ? ` on a ${rec.rootLength.toFixed(1)}mm root`
        + (rec.rootDefault != null ? ` (${fdiClass(rec.fdi)} default)` : " (session default)")
      : "";
    setStatus(`${who} selected${root} — gizmo resumed at its current position.`);
  };

  /**
   * The OTHER arch's session id, or null.
   *
   * Both arches live in the same raw scanner space — nothing is ever re-centred
   * (CLAUDE.md rule 3.1) — so an upper crown retracted lingually really does
   * land where the lower arch is. But they are separate SESSIONS on the server,
   * and nothing links them there, so the antagonist can only reach the
   * collision check if the client names it.
   *
   * DECLARED BEFORE applyClinical AND KEPT THERE. It appears in that callback's
   * dependency array, and a dependency array is evaluated DURING RENDER — not
   * when the callback runs. Declaring it below therefore crashed the whole app
   * with "Cannot access 'opposingSessionId' before initialization", a white
   * screen rather than a broken button, because the const had not initialised
   * yet when the array was built. Using it inside a callback BODY below would
   * have been fine; naming it in the deps is what forced the order.
   */
  const opposingSessionId = useCallback(() => {
    const other = active === "maxillary" ? "mandibular" : "maxillary";
    return sessions[other]?.session_id ?? null;
  }, [sessions, active]);

  /**
   * Sidebar -> tooth. The other half of the two-way binding.
   *
   * Values are ABSOLUTE from T0, so this is not an incremental nudge: the whole
   * prescription is rebuilt and applied. setClinical reproduces
   * cg.kinematic_matrix exactly (verified to 1.8e-15 by
   * frontend/verify-kinematics.mjs), so the viewport shows the same transform
   * the server will export.
   */
  const applyClinical = useCallback((key, value) => {
    const tid = activeTooth;
    const rec = teeth.current[tid];
    if (!rec || !gizmoRef.current?.pivot) return;

    // Base the new prescription on the RECORD, not on React state. rec.clinical
    // is updated synchronously here and by the gizmo's commit handler, whereas
    // `kinematics` only refreshes on re-render — so holding down "+" would
    // otherwise keep re-deriving from the same stale value and every click after
    // the first would be lost.
    const next = { ...(rec.clinical || kinematics), [key]: value };
    setKinematics(next);
    gizmoRef.current.setClinical(next);

    // Same persistence path a mouse drag takes — the gizmo is the single source
    // of the delta either way, so typed and dragged movements are indistinguishable
    // downstream.
    rec.delta.copy(gizmoRef.current.deltaMatrix());
    rec.clinical = next;
    // Typed movements move the tooth exactly as a drag does, so the shadow is
    // equally stale. Same flag, same reason.
    three.current.shadowsDirty = true;
    // A changed prescription changes the case length, and the timeline's own
    // stage count is what every stage pose is scaled against.
    const total = refreshStaging();
    setStage(total);

    const sid = sessions[active]?.session_id;
    if (sid) {
      fetch(`${API}/api/session/${sid}/tooth/${tid}/kinematics`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        // The opposing arch's session, when one is loaded. The two arches are
        // separate sessions and only the client knows both, so the antagonist
        // check cannot happen unless this is sent.
        body: JSON.stringify({ ...next, opposing_session_id: opposingSessionId() }),
      }).then(async (r) => {
        if (!r.ok) return;
        const d = await r.json();
        const warn = d.occlusal_warning
          ? ` ${d.occlusal_warning} (${d.occlusion.max_penetration_mm}mm into the antagonist.)`
          : "";
        rec.occlusion = d.occlusion || null;
        if (d.clearance?.over_threshold) {
          setStatus(`Interproximal contact: ${d.clearance.min_clearance_mm}mm clearance. `
                    + `${d.staging.stages_required} stages required.${warn}`);
        } else {
          setStatus(`${d.staging.stages_required} stages (${d.staging.driver}-driven).${warn}`);
        }
      }).catch(console.error);
    }
  }, [activeTooth, kinematics, sessions, active, opposingSessionId, refreshStaging]);

  const resetTooth = () => {
    const rec = teeth.current[activeTooth];
    if (!rec) return;
    rec.clinical = {tip_deg: 0, torque_deg: 0, rotation_deg: 0, d_md: 0, d_bl: 0, d_oa: 0};
    applyClinical("tip_deg", 0);          // one apply of the whole zero prescription
    refreshStaging();
  };

  // ===================================================================
  // Staging
  // ===================================================================

  const stagingRef = useRef(staging);
  stagingRef.current = staging;

  /**
   * Pose every committed crown at stage k. NOT a React render path.
   *
   * Writes matrices straight onto the meshes and detaches the gizmo while
   * scrubbing, because TransformControls would otherwise fight the poses it did
   * not author. Stage k is deltaFromClinical at clinical x k/N — absolute from
   * T0 — never a lerp of the committed 4x4: the 3x3 block of (1-t)I + tR is not
   * orthonormal for any t in between, so a matrix lerp shears and scales the
   * crown at every intermediate stage.
   */
  const applyStage = useCallback((k) => {
    const total = stagingRef.current.total;
    for (const rec of Object.values(teeth.current)) {
      if (!rec.mesh || !rec.frame) continue;
      const axes = rec.axes || (rec.axes = toothAxes(rec.frame));
      const at = total > 0 ? clinicalAtStage(rec.clinical, k, total) : rec.clinical;
      const M = deltaFromClinical(axes, rec.cRes, at);
      rec.mesh.matrixAutoUpdate = false;
      rec.mesh.matrix.copy(M);
      rec.mesh.matrixWorldNeedsUpdate = true;
      if (rec.rootMesh) {
        rec.rootMesh.matrixAutoUpdate = false;
        rec.rootMesh.matrix.copy(M);
        rec.rootMesh.matrixWorldNeedsUpdate = true;
      }
    }
    three.current.needsRender = true;
  }, []);

  const goToStage = useCallback((k) => {
    const total = stagingRef.current.total;
    const c = Math.max(0, Math.min(total, k));
    setStage(c);
    // The gizmo cannot stay attached while the timeline poses the scene — it
    // holds the crown at the committed pose and would fight every stage. detach
    // bakes the committed delta and hands the crown back to the scene with
    // matrixAutoUpdate off, which is exactly the state applyStage writes into.
    if (c !== total && gizmoRef.current?.pivot) {
      gizmoRef.current.detach();
      setActiveTooth(null);
    }
    applyStage(c);
  }, [applyStage]);

  const [playing, togglePlay] = useStagePlayback(staging.total, stage, goToStage);

  /**
   * Download the planned setup.
   *
   * This used to read a JSON body and set a status string. The server wrote
   * STLs into its own directory and returned their paths — no blob, no anchor,
   * no Content-Disposition, so nothing could ever reach the clinician's
   * Downloads folder however long they waited. The apparent hang was that plus
   * a long silence while cap_and_close sealed the base.
   *
   * The base is now trimmed to a horseshoe and extruded to a solid instead of
   * capped, which both removes the web cap_and_close stretched across the arch
   * opening and takes ~3s on a full arch rather than ~100.
   */
  /**
   * The manufacturing export: one fused solid per stage, ready to thermoform.
   *
   * Separate from exportSetup deliberately. That ships the planned setup — a
   * socketed base plus loose crowns, for inspection. This ships what a lab
   * actually pulls a sheet over: base and teeth FUSED, sockets filled flush,
   * one file per aligner.
   */
  /**
   * Point the shadow along the patient's apical direction and drop the catcher
   * under the cast. Called once the occlusal plane is established.
   */
  const aimShadows = useCallback((frame) => {
    const { sun, catcher } = three.current;
    if (!sun || !catcher || !frame?.u_occ) return;

    const up = new THREE.Vector3().fromArray(frame.u_occ).normalize();
    const box = new THREE.Box3();
    // rec is a plain record ({mesh, geometry, view, brushIndex, ...}), NOT an
    // Object3D — Box3.expandByObject calls updateWorldMatrix on its first line,
    // so passing rec threw a TypeError here and the whole shadow rig never armed.
    for (const rec of Object.values(arches.current)) if (rec?.mesh) box.expandByObject(rec.mesh);
    if (box.isEmpty()) return;
    const centre = box.getCenter(new THREE.Vector3());
    const radius = box.getSize(new THREE.Vector3()).length() * 0.5;

    // Lowest point of the cast along the patient's own axis, then a little
    // further, so the plane sits under the model rather than through it.
    const drop = up.clone().multiplyScalar(-(radius + 2));
    catcher.position.copy(centre).add(drop);
    catcher.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), up);
    catcher.visible = true;

    sun.target.position.copy(centre);
    sun.position.copy(centre).add(up.clone().multiplyScalar(radius * 3));
    sun.intensity = 1.1;
    const s = radius * 1.6;
    Object.assign(sun.shadow.camera, { left: -s, right: s, top: s, bottom: -s });
    sun.shadow.camera.updateProjectionMatrix();

    // Same record-vs-Object3D mistake as above: setting castShadow on the record
    // is a silent no-op, so the arch never cast a shadow. The teeth line below
    // always had it right.
    for (const rec of Object.values(arches.current)) if (rec?.mesh) rec.mesh.castShadow = true;
    for (const rec of Object.values(teeth.current)) if (rec.mesh) rec.mesh.castShadow = true;
    three.current.shadowsDirty = true;
  }, []);

  const exportStages = async () => {
    const sid = sessions[active]?.session_id;
    if (!sid || !staging.total) return;
    setBusy(true);
    // Say the number before the await; the boolean work is real.
    setStatus(`Fusing ${staging.total} stage models — each one a boolean union of `
              + `the cast and every crown. This takes a few seconds per stage...`);
    try {
      const res = await fetch(`${API}/api/session/${sid}/export/stages`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ opposing_session_id: opposingSessionId() }),
      });
      if (!res.ok) {
        const text = await res.text();
        let detail = text;
        try { detail = JSON.parse(text).detail ?? text; } catch { /* not JSON */ }
        throw new Error(detail);
      }
      const blob = await res.blob();
      const href = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = href;
      a.download = `${active}_stages.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(href);
      const kb = Math.round(blob.size / 1024);
      // The occlusion verdict rides in a header so the status line can say it
      // without the client unzipping the manifest it just handed to the user.
      const occ = res.headers.get("X-Occlusal-Interference");
      setStatus(`Downloaded ${a.download} (${kb.toLocaleString()} KB) — `
                + `${staging.total} fused stage models and manifest.json.`
                + (occ && occ !== "none"
                   ? `  Warning: occlusal interference with antagonist on stage(s) ${occ}.`
                   : ""));
    } catch (err) {
      setStatus(`Stage export failed: ${err.message}`);
    } finally { setBusy(false); }
  };

  const exportSetup = async () => {
    const sid = sessions[active]?.session_id;
    if (!sid) return;
    setBusy(true);
    // Say the number BEFORE the await. The silence is what reads as a hang.
    setStatus("Trimming the arch and extruding the cast base — a few seconds on a full arch...");
    try {
      const res = await fetch(`${API}/api/session/${sid}/export`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!res.ok) {
        const text = await res.text();
        let detail = text;
        try { detail = JSON.parse(text).detail ?? text; } catch { /* not JSON */ }
        // There is deliberately no "export anyway". The override that used to
        // live here existed to get past a base cap_and_close could not seal,
        // which was a misdiagnosis: the scan was a healthy open shell and the
        // capping was the defect. A trim-and-extrude base closes by
        // construction, so a refusal here is a bug to fix, not to wave through.
        throw new Error(detail);
      }

      const blob = await res.blob();
      const href = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = href;
      a.download = `${active}_setup.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(href);

      const kb = Math.round(blob.size / 1024);
      setStatus(`Downloaded ${a.download} (${kb.toLocaleString()} KB) — `
                + `solid cast base, one STL per tooth, and manifest.json.`);
    } catch (err) {
      setStatus(`Export failed: ${err.message}`);
    } finally { setBusy(false); }
  };

  /**
   * Hover highlight — an EMISSIVE BOOST on the material, not an OutlinePass.
   *
   * The brief asked for EffectComposer + OutlinePass, profiled, with an
   * emissive fallback if it missed 60fps. I could not run that profile: it
   * needs a real WebGL context and there is no browser in this environment, and
   * a number I cannot measure is not one I will report. So this ships the option
   * whose cost is known rather than guessed.
   *
   * What is known: OutlinePass adds a full-screen depth prepass, a separate
   * render of the selected objects to an offscreen target, and two blur passes
   * plus a composite — four extra passes over a ~200k-face arch, every frame,
   * whether anything is hovered or not. An emissive boost is a uniform change on
   * one material and costs nothing at all. On a clinical tool that has to stay
   * responsive while a gizmo is being dragged, that is the safer default.
   *
   * If OutlinePass is wanted, profile it in the browser first and keep this as
   * the fallback the brief already specified.
   */
  const setHover = useCallback((mesh) => {
    const prev = three.current.hovered;
    if (prev === mesh) return;

    // UNDO EXACTLY WHAT WAS DONE, which means recording which of the two
    // techniques was applied. The old code restored `emissiveIntensity`
    // unconditionally — including on a material that has no `emissive` and was
    // therefore never highlighted, where the write invents a property and
    // "restores" it to a 1 nobody stored.
    if (prev?.material) {
      const how = prev.userData.hoverTechnique;
      if (how === "emissive") {
        prev.material.emissive.setHex(prev.userData.prevEmissive ?? 0x000000);
        prev.material.emissiveIntensity = prev.userData.prevEmissiveI ?? 1;
      } else if (how === "color") {
        // Restore the SAVED hex, never `subScalar(0.12)`. addScalar clamps at
        // 1.0, so on a light material the boost is not invertible and hovering
        // repeatedly would walk the crown toward white a step at a time.
        prev.material.color.setHex(prev.userData.prevColor ?? 0xffffff);
      }
      prev.userData.hoverTechnique = null;
    }

    if (mesh?.material?.emissive) {
      mesh.userData.prevEmissive = mesh.material.emissive.getHex();
      mesh.userData.prevEmissiveI = mesh.material.emissiveIntensity;
      mesh.material.emissive.setHex(0x1e6fff);       // neon blue, as asked
      mesh.material.emissiveIntensity = 0.55;
      mesh.userData.hoverTechnique = "emissive";
    } else if (mesh?.material?.color) {
      // A material with no emissive channel — MeshBasicMaterial, or anything a
      // future renderer path substitutes. Silently doing nothing here is the
      // bad outcome: the clinician gets no feedback that the tooth under the
      // cursor is the one that will be picked, and there is no way to tell
      // that apart from a dead pointer handler.
      mesh.userData.prevColor = mesh.material.color.getHex();
      mesh.material.color.addScalar(0.12);
      mesh.userData.hoverTechnique = "color";
    }

    three.current.hovered = mesh || null;
  }, []);

  const onPointerMove = (e) => {
    if (pickMode || busy) return;
    const st = stateRef.current;
    if (!st?.camera) return;
    const crowns = Object.values(teeth.current).map((t) => t.mesh).filter(Boolean);
    if (!crowns.length) { setHover(null); return; }
    const rect = e.currentTarget.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1);
    st.raycaster.setFromCamera(ndc, st.camera);
    const hit = st.raycaster.intersectObjects(crowns, false)[0];
    setHover(hit?.object || null);
  };

  const onPointerDown = async (e) => {
    // Cut crowns are tested BEFORE the arch: they sit proud of the cast, and a
    // click meant for a tooth that has been moved would otherwise fall through
    // to the gingiva behind it and start a wand selection.
    if (!pickMode && tool === "wand" && !e.shiftKey && !e.altKey && e.button === 0) {
      const rect = mountRef.current.getBoundingClientRect();
      const ndc = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1);
      three.current.raycaster.setFromCamera(ndc, three.current.camera);
      const crowns = Object.values(teeth.current).map((t) => t.mesh);
      const crownHit = three.current.raycaster.intersectObjects(crowns, false)[0];
      if (crownHit) { selectTooth(crownHit.object.userData.toothId); return; }
    }

    const hit = pickAcrossArches(e, mountRef.current, three.current.camera, three.current.raycaster, arches.current);

    if (pickMode && hit) {
      const pt = [hit.point.x, hit.point.y, hit.point.z];
      if (pickMode === "mesial") setMesialPt(pt);
      if (pickMode === "distal") setDistalPt(pt);
      setPickMode(null);
      setStatus(`${pickMode} point set!`);
      return;
    }

    const painting = tool !== "wand" || e.shiftKey || e.altKey;
    if (painting || e.button !== 0) return; 

    if (!hit) return;
    const sid = sessions[hit.archName]?.session_id;
    if (!sid) return;
    
    setActive(hit.archName);
    setStatus("Calculating boundaries...");
    
    const res = await fetch(`${API}/api/session/${sid}/wand`, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({point: [hit.point.x, hit.point.y, hit.point.z]}) 
    });
    
    if (res.ok) {
      const data = await res.json();
      setTolerance(data.tolerance); 
      setSelection(data.vertex_ids);
      highlightSelection(arches.current[hit.archName].geometry, data.vertex_ids);
      setStatus(`Selected ${data.vertex_ids.length} vertices. (Auto-tolerance: ${data.tolerance.toFixed(1)})`);
      
      fetch(`${API}/api/session/${sid}/selection`, {
        method:"PUT", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({vertex_ids: data.vertex_ids})
      });
    }
  };

  const updateThreshold = async (val) => {
    setTolerance(val);
    const sid = sessions[active]?.session_id;
    if (!sid) return;
    const res = await fetch(`${API}/api/session/${sid}/wand/threshold`, {
        method:"POST", headers:{"Content-Type":"application/json"},
        body: JSON.stringify({tolerance: val})
    });
    if (res.ok) {
        const data = await res.json();
        setSelection(data.vertex_ids);
        highlightSelection(arches.current[active].geometry, data.vertex_ids);
    }
  };

  return (
    <div style={S.app}>
      <aside style={S.rail}>
        <PanelGroup defaultOpen={["biometrics", "prep", "kinematics", "export"]}>
        <h1 style={S.titleTight}>Clinical Micro-Planner</h1>

        <div style={{ ...S.health, borderColor: health.state === "offline" ? "#6b1f2e" : "#2c313a" }}
             title={`${API} — ${HEALTH_UI[health.state].label}`}>
          <span style={{ ...S.healthDot, background: HEALTH_UI[health.state].dot }} />
          <span>
            {HEALTH_UI[health.state].label}
            {health.detail ? <span style={{ color: "#8b93a0" }}> — {health.detail}</span> : null}
          </span>
        </div>

        <Panel id="biometrics" step="1" title="Arches & Biometrics">
          {["maxillary", "mandibular"].map((a) => (
            <div key={a} style={{ marginBottom: 8 }}>
              <label style={S.button}>
                Load {a} STL
                <input type="file" accept=".stl,.obj" disabled={busy} onChange={(e) => loadArch(e, a)} style={{ display: "none" }} />
              </label>
            </div>
          ))}
          <button onClick={handleDefineOcclusalPlane} disabled={!sessions[active] || busy} style={{...S.chip, marginTop: 8, borderColor: archFrame ? "#3cb44b" : "#2c313a"}}>
            {archFrame ? "✓ Occlusal Plane Set" : "Define Occlusal Plane (3 Clicks)"}
          </button>
          <button onClick={runSegmentation} disabled={busy || segmenting || !sessions[active]} style={{...S.primary, marginTop: 10}}>
            {segmenting ? "Segmenting…" : "Segment Teeth"}
          </button>
        </Panel>

        <Panel id="prep" step="2" title="Crown Prep — Selection">
          <div style={{display:"flex", gap:6, marginBottom: 10}}>
            {[["wand","Wand"],["brush","Brush +"],["erase","Brush −"]].map(([k,label]) => (
              <button key={k} onClick={() => setTool(k)}
                style={{...S.button, flex:1, fontSize: 11, padding: "6px",
                        borderColor: tool===k ? "#3fc6d4" : "#2c313a",
                        color: tool===k ? "#3fc6d4" : "#e7ebee"}}>{label}</button>
            ))}
          </div>
          {tool === "wand" ? (
            <>
              <div style={S.dt}>Tolerance ({tolerance.toFixed(1)})</div>
              <input type="range" min={0.5} max={15.0} step={0.1} value={tolerance}
                     onChange={(e) => updateThreshold(Number(e.target.value))}
                     style={{ width: "100%", accentColor: "#3fc6d4" }} />
            </>
          ) : (
            <>
              <div style={S.dt}>Brush Radius ({radius.toFixed(1)} mm)</div>
              <input type="range" min={0.5} max={5.0} step={0.1} value={radius}
                     onChange={(e) => setRadius(Number(e.target.value))}
                     style={{ width: "100%", accentColor: "#3fc6d4" }} />
            </>
          )}
        </Panel>

        <Panel id="cut" step="3" title="Crown Prep — Clinical Cut">
          {(() => {
            // Show the root length that will ACTUALLY be used for this
            // selection, and where it came from. A clinician overriding a
            // number needs to see what it was derived from; a bare slider
            // reading 10.0 for a canine looks deliberate and is not.
            const fdi = fdiForSelection(selection);
            const derived = fdi == null ? null : rootDefaultForFDI(fdi);
            return derived == null ? (
              <>
                <div style={S.dt}>Virtual Root Length ({rootLength.toFixed(1)} mm)</div>
                <input type="range" min={7} max={16} step={0.5} value={rootLength}
                       onChange={(e) => setRootLength(Number(e.target.value))}
                       style={{ width: "100%", accentColor: "#3fc6d4" }} />
                <div style={{...S.dt, color: "#8b93a0", marginBottom: 10}}>
                  Session default — run segmentation to derive it per tooth.
                </div>
              </>
            ) : (
              <div style={{marginBottom: 10}}>
                <div style={S.dt}>Virtual Root Length ({derived.toFixed(1)} mm)</div>
                <div style={{...S.dt, color: "#3fc6d4"}}>
                  FDI {fdi} — {fdiClass(fdi)} default,
                  {" "}{ROOT_DEFAULTS_MM[fdiClass(fdi)]} mm (Wheeler)
                </div>
              </div>
            );
          })()}

          <button onClick={() => setPickMode("mesial")} style={{...S.chip, borderColor: mesialPt ? "#3cb44b" : (pickMode === "mesial" ? "#3fa9ff" : "#2c313a")}}>
            {mesialPt ? "✓ Mesial Set" : (pickMode === "mesial" ? "Click mesh..." : "Set Mesial Point")}
          </button>
          <button onClick={() => setPickMode("distal")} style={{...S.chip, borderColor: distalPt ? "#3cb44b" : (pickMode === "distal" ? "#3fa9ff" : "#2c313a")}}>
            {distalPt ? "✓ Distal Set" : (pickMode === "distal" ? "Click mesh..." : "Set Distal Point")}
          </button>
          <button onClick={executeCut} disabled={!mesialPt || !distalPt || !selection.length || busy} style={{...S.primary, marginTop: 10, background: (!mesialPt || !distalPt) ? "#2a2f37" : "linear-gradient(#3fc6d4,#2a8b96)"}}>
            Extract Crown
          </button>
        </Panel>

        {activeTooth && (
          <Panel id="kinematics" step="4" title="Kinematics">
            <div style={{display: "flex", gap: 6, marginBottom: 10}}>
              <button onClick={() => { setGizmoMode("rotate"); gizmoRef.current?.setMode("rotate"); }}
                      style={{...S.button, flex: 1, borderColor: gizmoMode === "rotate" ? "#3fc6d4" : "#2c313a"}}>
                Rotate
              </button>
              <button onClick={() => { setGizmoMode("translate"); gizmoRef.current?.setMode("translate"); }}
                      style={{...S.button, flex: 1, borderColor: gizmoMode === "translate" ? "#3fc6d4" : "#2c313a"}}>
                Translate
              </button>
            </div>
            
            {CONTROLS.map((spec) => (
              <ClinicalInput key={spec.key} spec={spec}
                             value={Number(kinematics[spec.key]) || 0}
                             disabled={busy || !activeTooth}
                             onCommit={(v) => applyClinical(spec.key, v)} />
            ))}

            <button onClick={resetTooth} disabled={busy || !activeTooth}
                    style={{...S.chip, marginTop: 4}}>
              Reset to T0
            </button>
          </Panel>
        )}

        <ValidationPanel checks={caseChecks} />

        {Object.keys(teeth.current).length > 0 && (
          <>
          <Panel id="attachments" step="6" title="Attachments">
            <AttachmentPanel
              enabled={!!activeTooth}
              active={attachMode}
              onToggle={() => setAttachMode((m) => !m)}
              settings={attachSettings}
              onChange={setAttachSettings}
              placed={placedAttachments}
              onClear={clearAttachments}
              targetLabel={activeTooth
                ? (teeth.current[activeTooth]?.fdi
                    ? `FDI ${teeth.current[activeTooth].fdi}`
                    : activeTooth.slice(0, 8))
                : null} />
          </Panel>

          <Panel id="export" step="7" title="Export">
            {Object.entries(teeth.current).map(([tid, rec]) => {
              const moved = rec.clinical && Object.values(rec.clinical).some((x) => x);
              return (
                <button key={tid} onClick={() => selectTooth(tid)}
                        style={{...S.chip, textAlign: "left",
                                borderColor: activeTooth === tid ? "#3fc6d4" : "#2c313a",
                                color: activeTooth === tid ? "#3fc6d4" : "#8b93a0"}}>
                  {activeTooth === tid ? "● " : "○ "}
                  {rec.fdi != null ? `FDI ${rec.fdi}` : tid}
                  {rec.rootLength != null ? `  ${rec.rootLength.toFixed(1)}mm root` : ""}
                  {moved ? "  (moved)" : "  (at T0)"}
                  {rec.rootClamp ? (
                    <div style={S.rootClamp} title={rec.rootClamp.clamp_note}>
                      [Virtual Root] Projection constrained to alveolar boundary
                      {" — "}
                      {rec.rootClamp.requested_mm.toFixed(1)}mm requested,{" "}
                      {rec.rootClamp.length_mm.toFixed(1)}mm drawn. Pivot unchanged.
                    </div>
                  ) : null}
                </button>
              );
            })}
            <button onClick={exportSetup} disabled={busy}
                    style={{...S.primary, marginTop: 10,
                            background: "linear-gradient(#3fc6d4,#2a8b96)"}}>
              Export Printable Cast
            </button>
            <button onClick={exportStages} disabled={busy || !staging.total}
                    style={{...S.chip, marginTop: 6}}>
              Export Stages (1&ndash;{staging.total || "?"}) &mdash; fused solids
            </button>
          </Panel>
          </>
        )}
        </PanelGroup>
      </aside>
      <main ref={mountRef} style={{...S.canvas, cursor: pickMode ? "crosshair" : (tool === "wand" ? "crosshair" : "cell")}} onPointerDown={onPointerDown} onPointerMove={onPointerMove}>
        <StagingTimeline totalStages={staging.total} stage={stage} playing={playing}
                         perTooth={staging.perTooth}
                         onStage={goToStage} onPlayPause={togglePlay} />
      </main>
      <footer style={S.status}>{status}</footer>
    </div>
  );
}

const S = {
  app: { display: "grid", gridTemplateColumns: "300px 1fr", gridTemplateRows: "1fr auto", height: "100vh", background: "#0d0f12", color: "#e7ebee", fontFamily: "system-ui, sans-serif", fontSize: 13 },
  rail: { gridRow: "1 / 3", background: "#1a1d22", borderRight: "1px solid #2c313a", padding: 16, overflowY: "auto" },
  title: { fontSize: 15, fontWeight: 600, margin: "0 0 18px" },
  titleTight: { fontSize: 15, fontWeight: 600, margin: "0 0 10px" },
  health: { display: "flex", alignItems: "center", gap: 8, padding: "6px 9px", marginBottom: 16,
            borderRadius: 5, background: "#101215", border: "1px solid #2c313a",
            fontSize: 11, lineHeight: 1.35, color: "#c3cad3" },
  healthDot: { width: 8, height: 8, borderRadius: "50%", flexShrink: 0 },
  group: { border: "1px solid #2c313a", borderRadius: 10, padding: 12, marginBottom: 12, background: "rgba(255,255,255,0.02)" },
  legend: { color: "#3fc6d4", fontSize: 10, fontWeight: 700, letterSpacing: 1.2, marginBottom: 10 },
  button: { display: "block", background: "linear-gradient(#2a2f37,#21252b)", border: "1px solid #2c313a", borderRadius: 6, padding: "8px 12px", cursor: "pointer", textAlign: "center" },
  primary: { width: "100%", color: "#06181a", border: "none", borderRadius: 6, padding: "8px 12px", fontWeight: 600, cursor: "pointer" },
  chip: { width: "100%", marginTop: 5, background: "transparent", color: "#8b93a0", border: "1px solid #2c313a", borderRadius: 5, padding: "6px 8px", fontSize: 12, cursor: "pointer", transition: "0.2s" },
  dt: { color: "#8b93a0", marginBottom: 2, marginTop: 8 },
  // Amber, not red: the plan is fine, the DRAWING was shortened. A red
  // chip here would read as a clinical finding about the tooth.
  rootClamp: { marginTop: 5, padding: "4px 6px", borderRadius: 4,
               background: "#2a2113", border: "1px solid #5a4620",
               color: "#ffa53c", fontSize: 10.5, lineHeight: 1.45,
               whiteSpace: "normal" },
  stepBtn: {
    width: 28, flex: "0 0 28px", background: "linear-gradient(#2a2f37,#21252b)",
    border: "1px solid #2c313a", borderRadius: 5, color: "#e7ebee",
    fontSize: 15, lineHeight: 1, cursor: "pointer", padding: 0,
  },
  inputWrap: {
    flex: 1, display: "flex", alignItems: "center", background: "#101215",
    border: "1px solid #2c313a", borderRadius: 5, padding: "0 6px", minWidth: 0,
  },
  numInput: {
    flex: 1, minWidth: 0, width: "100%", background: "transparent", border: "none",
    outline: "none", color: "#fff", fontSize: 13, padding: "5px 0",
    fontVariantNumeric: "tabular-nums",   // digits stop jittering as values change
    textAlign: "right", fontFamily: "inherit",
  },
  unit: { color: "#8b93a0", fontSize: 11, marginLeft: 4 },
  canvas: { position: "relative", overflow: "hidden", userSelect: "none", touchAction: "none" },
  status: { background: "#101215", borderTop: "1px solid #2c313a", color: "#8b93a0", padding: "7px 14px" },
};