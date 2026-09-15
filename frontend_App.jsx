import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const API = "http://127.0.0.1:8000";

const GINGIVA = new THREE.Color("#d9a7a0");
const SELECTED = new THREE.Color("#3fa9ff");
const PALETTE = ["#f2ede0", "#e8e0c8", "#f5efe2", "#ded6bd", "#efe7d4",
                 "#e3dcc4", "#f7f1e4", "#d9d1b8"].map((c) => new THREE.Color(c));

export default function App() {
  const mountRef = useRef(null);
  const three = useRef({});
  const pointerDown = useRef(null);
  const arches = useRef({});          // { maxillary: {...}, mandibular: {...} }

  const [status, setStatus] = useState("Load an arch to begin.");
  const [sessions, setSessions] = useState({});
  const [active, setActive] = useState("maxillary");
  const [tolerance, setTolerance] = useState(10);
  const [separation, setSeparation] = useState(5.5);
  const [selection, setSelection] = useState(null);
  const [autoInfo, setAutoInfo] = useState(null);
  const [seed, setSeed] = useState(null);
  const [busy, setBusy] = useState(false);
  const [brushRadius, setBrushRadius] = useState(1.5);
  const [brushMode, setBrushMode] = useState(null);   // "add" | "remove" | null
  const [painted, setPainted] = useState(0);
  const selMask = useRef({});      // archName -> Uint8Array, one byte per face
  const brushing = useRef(false);

  // ---------------------------------------------------------------- scene
  useEffect(() => {
    const mount = mountRef.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d0f12);

    const camera = new THREE.PerspectiveCamera(
      45, mount.clientWidth / mount.clientHeight, 0.1, 5000);
    camera.position.set(0, -80, 60);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 0.8);
    key.position.set(1, -1, 1); scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.35);
    fill.position.set(-1, 1, 0.5); scene.add(fill);

    three.current = { scene, camera, renderer, controls, raycaster: new THREE.Raycaster() };

    let raf;
    (function animate() {
      raf = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    })();

    const onResize = () => {
      if (!mount.clientWidth) return;
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
    };
    window.addEventListener("resize", onResize);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      controls.dispose(); renderer.dispose();
      if (renderer.domElement.parentNode) mount.removeChild(renderer.domElement);
    };
  }, []);

  // ---------------------------------------------------- load a given arch
  const loadArch = useCallback(async (event, archName) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setBusy(true);
    setStatus(`Reading ${file.name} as ${archName}...`);

    try {
      const buffer = await file.arrayBuffer();
      const geometry = new STLLoader().parse(buffer);
      geometry.computeVertexNormals();
      // No centring, no scaling: both arches must stay in raw scanner
      // coordinates or the bite registration between them is destroyed, and
      // every click sent to the backend would be in the wrong frame.

      // STLLoader returns NON-INDEXED geometry: face i owns vertices
      // 3i, 3i+1, 3i+2. That makes per-face colouring trivial, and the
      // numbering matches the backend's face indices exactly (verified in
      // test_face_order.py), so indices need no translation.
      const nVerts = geometry.attributes.position.count;
      const colors = new Float32Array(nVerts * 3);
      for (let i = 0; i < nVerts; i++) GINGIVA.toArray(colors, i * 3);
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

      const { scene, camera, controls } = three.current;
      if (arches.current[archName]) {
        scene.remove(arches.current[archName].mesh);
        arches.current[archName].mesh.geometry.dispose();
      }

      const mesh = new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({
        vertexColors: true, specular: 0x222222, shininess: 22,
        side: THREE.DoubleSide,
      }));
      scene.add(mesh);
      // Face centroids, computed once. The brush needs to find every face
      // within a radius of the cursor on every pointermove; recomputing
      // centroids per event would be hopeless on a 220k-face scan, but a
      // flat Float32Array scanned linearly is a couple of milliseconds.
      const pos = geometry.attributes.position.array;
      const nFaces = nVerts / 3;
      const centroids = new Float32Array(nFaces * 3);
      for (let f = 0; f < nFaces; f++) {
        const o = f * 9;
        centroids[f * 3]     = (pos[o]     + pos[o + 3] + pos[o + 6]) / 3;
        centroids[f * 3 + 1] = (pos[o + 1] + pos[o + 4] + pos[o + 7]) / 3;
        centroids[f * 3 + 2] = (pos[o + 2] + pos[o + 5] + pos[o + 8]) / 3;
      }
      arches.current[archName] = { mesh, geometry, nFaces, centroids, baseColors: null };
      selMask.current[archName] = new Uint8Array(nFaces);

      // frame the camera on everything currently loaded
      const box = new THREE.Box3();
      Object.values(arches.current).forEach((a) => box.expandByObject(a.mesh));
      const sphere = box.getBoundingSphere(new THREE.Sphere());
      controls.target.copy(sphere.center);
      camera.position.set(sphere.center.x,
                          sphere.center.y - sphere.radius * 2.2,
                          sphere.center.z + sphere.radius * 1.2);
      camera.near = sphere.radius / 100;
      camera.far = sphere.radius * 50;
      camera.updateProjectionMatrix();
      controls.update();

      const form = new FormData();
      form.append("file", file);
      const res = await fetch(`${API}/session?arch=${archName}`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
      const info = await res.json();
      setSessions((s) => ({ ...s, [archName]: info }));
      setActive(archName);
      setStatus(`${archName}: ${info.n_faces.toLocaleString()} faces, prep ${info.prep_seconds}s. ` +
                `Click a tooth, or run Auto-colour.`);
    } catch (err) {
      setStatus(`Failed: ${err.message}. Is the backend running? (uvicorn server:app --reload)`);
    } finally {
      setBusy(false);
    }
  }, []);

  // ------------------------------------------------------------ colouring
  // Repaint a specific set of faces from the selection mask. Only the faces
  // that changed are written, so a brush stroke costs a few hundred vertex
  // updates rather than a full re-colour of the arch.
  const repaintFaces = useCallback((archName, faceList) => {
    const arch = arches.current[archName];
    if (!arch) return;
    const attr = arch.geometry.attributes.color;
    const mask = selMask.current[archName];
    const base = arch.baseColors;
    for (const f of faceList) {
      const col = mask[f] ? SELECTED
        : (base ? null : GINGIVA);
      for (let c = 0; c < 3; c++) {
        const o = (f * 3 + c) * 3;
        if (col) col.toArray(attr.array, o);
        else { attr.array[o] = base[o]; attr.array[o+1] = base[o+1]; attr.array[o+2] = base[o+2]; }
      }
    }
    attr.needsUpdate = true;
  }, []);

  const setSelectionFromIndices = useCallback((archName, indices) => {
    const arch = arches.current[archName];
    if (!arch) return;
    const mask = selMask.current[archName];
    const changed = [];
    for (let f = 0; f < mask.length; f++) if (mask[f]) { mask[f] = 0; changed.push(f); }
    for (const f of indices) { mask[f] = 1; changed.push(f); }
    repaintFaces(archName, changed);
    setPainted(indices.length);
  }, [repaintFaces]);

  const applyLabels = useCallback((archName, labels) => {
    const arch = arches.current[archName];
    if (!arch) return;
    const attr = arch.geometry.attributes.color;
    for (let f = 0; f < labels.length; f++) {
      const col = labels[f] === 0 ? GINGIVA : PALETTE[(labels[f] - 1) % PALETTE.length];
      for (let c = 0; c < 3; c++) col.toArray(attr.array, (f * 3 + c) * 3);
    }
    attr.needsUpdate = true;
    arch.baseColors = Float32Array.from(attr.array);   // selections restore to this
  }, []);

  // --------------------------------------------------- POST /auto-color
  const runAutoColor = useCallback(async () => {
    const sid = sessions[active]?.session_id;
    if (!sid) { setStatus("Load that arch first."); return; }
    setBusy(true); setStatus("Detecting teeth across the arch...");
    try {
      const res = await fetch(`${API}/session/${sid}/auto-color`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tolerance: 12.0, min_separation_mm: separation, max_seeds: 16 }),
      });
      if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
      const meta = JSON.parse(res.headers.get("X-Meta"));
      // int16 labels, one per face, raw little-endian - no zip or npy parsing
      const labels = new Int16Array(await res.arrayBuffer());
      applyLabels(active, labels);
      setAutoInfo(meta);
      setStatus(`Auto-colour: ${meta.n_teeth_found} teeth, ` +
                `${(meta.tooth_fraction * 100).toFixed(1)}% of arch. ` +
                `Adjust separation if teeth merged or split.`);
    } catch (err) {
      setStatus(`Auto-colour failed: ${err.message}`);
    } finally { setBusy(false); }
  }, [sessions, active, separation, applyLabels]);

  // ------------------------------------------------------ POST /preview
  // Takes the arch name explicitly rather than reading `active`. React state
  // updates are asynchronous, so immediately after setActive(other) the
  // `active` closure still holds the OLD arch -- the preview would be
  // computed against the wrong session.
  const runPreviewOn = useCallback(async (archName, seedPoint, tol) => {
    const sid = sessions[archName]?.session_id;
    if (!sid) return;
    try {
      const res = await fetch(`${API}/session/${sid}/preview`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed: seedPoint, tolerance: tol }),
      });
      if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
      const n = Number(res.headers.get("X-Selected-Faces"));
      const frac = Number(res.headers.get("X-Selected-Fraction"));
      const plausible = res.headers.get("X-Plausible") === "True";
      const indices = new Int32Array(await res.arrayBuffer());
      setSelectionFromIndices(archName, indices);
      setSelection({ n, frac, plausible });
      setStatus(`Selected ${n.toLocaleString()} faces (${(frac * 100).toFixed(1)}% of arch)` +
                (plausible ? " - plausible for one crown."
                           : " - outside the usual 2-10%; adjust spread."));
    } catch (err) {
      setStatus(`Preview failed: ${err.message}`);
    }
  }, [sessions, setSelectionFromIndices]);

  // Re-threshold as the slider moves. The backend caches the distance field
  // per seed, so this is an array threshold rather than a fresh graph search.
  useEffect(() => {
    if (seed) runPreviewOn(active, seed, tolerance);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tolerance]);

  // --------------------------------------------------------------- brush
  // Shift+drag paints faces into the selection, Alt+drag erases. Entirely
  // local: no request per stroke, so the highlight tracks the cursor.
  // The backend only sees the result, when Confirm sends the face list.
  const applyBrush = useCallback((archName, point, mode) => {
    const arch = arches.current[archName];
    if (!arch) return;
    const mask = selMask.current[archName];
    const c = arch.centroids;
    const r2 = brushRadius * brushRadius;
    const want = mode === "add" ? 1 : 0;
    const changed = [];
    for (let f = 0; f < arch.nFaces; f++) {
      if (mask[f] === want) continue;
      const dx = c[f*3] - point.x, dy = c[f*3+1] - point.y, dz = c[f*3+2] - point.z;
      if (dx*dx + dy*dy + dz*dz <= r2) { mask[f] = want; changed.push(f); }
    }
    if (changed.length) {
      repaintFaces(archName, changed);
      let total = 0;
      for (let f = 0; f < mask.length; f++) total += mask[f];
      setPainted(total);
    }
  }, [brushRadius, repaintFaces]);

  const rayHit = useCallback((clientX, clientY, meshes) => {
    const { raycaster, camera, renderer } = three.current;
    const rect = renderer.domElement.getBoundingClientRect();
    raycaster.setFromCamera(new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1), camera);
    return raycaster.intersectObjects(meshes, false)[0];
  }, []);

  const onPointerMove = (e) => {
    if (!brushing.current) return;
    const arch = arches.current[active];
    if (!arch) return;
    const hit = rayHit(e.clientX, e.clientY, [arch.mesh]);
    if (hit) applyBrush(active, hit.point, brushing.current);
  };

  // ------------------------------------------------------------- picking
  // OrbitControls and picking both want the left button - the same conflict
  // that broke the desktop build. Treat it as a click only if the pointer
  // barely moved between down and up.
  const DRAG_PX = 6;
  const onPointerDown = (e) => {
    pointerDown.current = { x: e.clientX, y: e.clientY };
    // Disable OrbitControls while brushing, otherwise the drag rotates the
    // camera underneath the stroke.
    if (e.shiftKey || e.altKey) {
      brushing.current = e.altKey ? "remove" : "add";
      setBrushMode(brushing.current);
      three.current.controls.enabled = false;
      const arch = arches.current[active];
      const hit = arch && rayHit(e.clientX, e.clientY, [arch.mesh]);
      if (hit) applyBrush(active, hit.point, brushing.current);
    }
  };

  const onPointerUp = (e) => {
    const start = pointerDown.current;
    pointerDown.current = null;
    if (brushing.current) {
      brushing.current = false;
      setBrushMode(null);
      three.current.controls.enabled = true;
      setStatus(`Painted selection: ${painted.toLocaleString()} faces.`);
      return;
    }
    if (!start) return;
    const dx = e.clientX - start.x, dy = e.clientY - start.y;
    if (dx * dx + dy * dy > DRAG_PX * DRAG_PX) return;

    const { raycaster, camera, renderer } = three.current;
    const rect = renderer.domElement.getBoundingClientRect();
    raycaster.setFromCamera(new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1), camera);

    // Raycast against EVERY loaded arch, not just the active one.
    //
    // This was the "nothing under cursor" bug, and it was not a stale-state
    // problem: `active` was always current. The previous code called
    // intersectObject(activeArch) and so was structurally incapable of
    // seeing the other arch. Clicking the maxilla while the mandible was
    // active tested the ray against the mandible only -- returning either no
    // hit, or worse, a silent hit on the mandible hidden behind the maxilla,
    // which would have sent a seed point on the wrong arch to the backend.
    //
    // Testing all arches and taking the nearest hit also gives the right
    // behaviour for free: clicking an arch selects that arch. The uuid maps
    // the hit object back to its name, since Three.js returns the Mesh, not
    // our wrapper.
    const meshes = Object.values(arches.current).map((a) => a.mesh);
    if (!meshes.length) return;
    const hit = raycaster.intersectObjects(meshes, false)[0];
    if (!hit) { setStatus("No surface under the cursor."); return; }

    const clickedArch = Object.keys(arches.current).find(
      (name) => arches.current[name].mesh.uuid === hit.object.uuid);
    if (!clickedArch) return;

    if (clickedArch !== active) {
      // Switching arches invalidates the previous seed: it is a coordinate
      // on a different mesh, and the backend session it belongs to is a
      // different session entirely.
      setActive(clickedArch);
      setSelection(null);
      setStatus(`Switched to ${clickedArch}.`);
    }

    const p = [hit.point.x, hit.point.y, hit.point.z];
    setSeed(p);
    runPreviewOn(clickedArch, p, tolerance);
  };

  // ----------------------------------------------------------------- UI
  return (
    <div style={S.app}>
      <aside style={S.rail}>
        <h1 style={S.title}>Clinical Micro-Planner</h1>

        <section style={S.group}>
          <div style={S.legend}>ARCHES</div>
          {["maxillary", "mandibular"].map((a) => (
            <div key={a} style={{ marginBottom: 8 }}>
              <label style={S.button}>
                {sessions[a] ? `Reload ${a}` : `Load ${a} STL`}
                <input type="file" accept=".stl" disabled={busy}
                       onChange={(e) => loadArch(e, a)} style={{ display: "none" }} />
              </label>
              {sessions[a] && (
                <button onClick={() => { setActive(a); setSeed(null); setSelection(null); }}
                        style={{ ...S.chip, ...(active === a ? S.chipOn : {}) }}>
                  {active === a ? "active" : "set active"} - {sessions[a].n_faces.toLocaleString()} faces
                </button>
              )}
            </div>
          ))}
        </section>

        <section style={S.group}>
          <div style={S.legend}>AUTO-COLOUR</div>
          <Slider label="Tooth separation" value={separation} min={3} max={12} step={0.1}
                  suffix="mm" onChange={setSeparation} />
          <button onClick={runAutoColor} disabled={busy || !sessions[active]}
                  style={S.primary}>Detect teeth</button>
          {autoInfo && (
            <dl style={S.dl}>
              <Row k="teeth found" v={autoInfo.n_teeth_found} />
              <Row k="tooth area" v={`${(autoInfo.tooth_fraction * 100).toFixed(1)}%`} />
            </dl>
          )}
          <div style={S.warn}>
            Visual guide only. Verify every boundary before cutting.
          </div>
        </section>

        <section style={S.group}>
          <div style={S.legend}>BRUSH</div>
          <Slider label="Brush radius" value={brushRadius} min={0.3} max={5} step={0.1}
                  suffix="mm" onChange={setBrushRadius} />
          <div style={S.muted}>
            <b>Shift + drag</b> to add, <b>Alt + drag</b> to erase.
          </div>
          {painted > 0 && (
            <dl style={S.dl}><Row k="painted faces" v={painted.toLocaleString()} /></dl>
          )}
          {brushMode && <div style={{ ...S.dd, marginTop: 6 }}>brushing: {brushMode}</div>}
        </section>

        <section style={S.group}>
          <div style={S.legend}>MAGIC WAND</div>
          <Slider label="Selection spread" value={tolerance} min={1} max={40} step={0.5}
                  suffix="mm" onChange={setTolerance} />
          {selection ? (
            <dl style={S.dl}>
              <Row k="faces" v={selection.n.toLocaleString()} />
              <Row k="fraction" v={`${(selection.frac * 100).toFixed(1)}%`}
                   accent={selection.plausible ? undefined : "#e0a05a"} />
            </dl>
          ) : <div style={S.muted}>Click a tooth to select it.</div>}
        </section>
      </aside>

      <main ref={mountRef} style={S.canvas}
            onPointerDown={onPointerDown} onPointerUp={onPointerUp}
            onPointerMove={onPointerMove} onPointerLeave={onPointerUp} />
      <footer style={S.status}>{status}</footer>
    </div>
  );
}

function Slider({ label, value, min, max, step, suffix, onChange }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={S.row}>
        <span style={S.dt}>{label}</span>
        <span style={S.dd}>{value}{suffix}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
             onChange={(e) => onChange(Number(e.target.value))}
             style={{ width: "100%", accentColor: "#3fc6d4" }} />
    </div>
  );
}

function Row({ k, v, accent }) {
  return (
    <div style={S.row}>
      <dt style={S.dt}>{k}</dt>
      <dd style={{ ...S.dd, color: accent || S.dd.color }}>{v}</dd>
    </div>
  );
}

const ACCENT = "#3fc6d4";
const S = {
  app: { display: "grid", gridTemplateColumns: "300px 1fr", gridTemplateRows: "1fr auto",
         height: "100vh", background: "#0d0f12", color: "#e7ebee",
         fontFamily: "system-ui, -apple-system, Segoe UI, sans-serif", fontSize: 13 },
  rail: { gridRow: "1 / 3", background: "#1a1d22", borderRight: "1px solid #2c313a",
          padding: 16, overflowY: "auto" },
  title: { fontSize: 15, fontWeight: 600, margin: "0 0 18px" },
  group: { border: "1px solid #2c313a", borderRadius: 10, padding: 12, marginBottom: 12,
           background: "rgba(255,255,255,0.02)" },
  legend: { color: ACCENT, fontSize: 10, fontWeight: 700, letterSpacing: 1.2, marginBottom: 10 },
  button: { display: "block", background: "linear-gradient(#2a2f37,#21252b)",
            border: "1px solid #2c313a", borderRadius: 6, padding: "8px 12px",
            cursor: "pointer", textAlign: "center" },
  primary: { width: "100%", background: "linear-gradient(#3fc6d4,#2a8b96)", color: "#06181a",
             border: "none", borderRadius: 6, padding: "8px 12px", fontWeight: 600,
             cursor: "pointer" },
  chip: { width: "100%", marginTop: 5, background: "transparent", color: "#8b93a0",
          border: "1px solid #2c313a", borderRadius: 5, padding: "4px 8px",
          fontSize: 11, cursor: "pointer" },
  chipOn: { color: ACCENT, borderColor: ACCENT },
  muted: { color: "#8b93a0", lineHeight: 1.5 },
  warn: { color: "#8b93a0", fontSize: 11, marginTop: 8, lineHeight: 1.45,
          borderLeft: "2px solid #e0a05a", paddingLeft: 8 },
  dl: { margin: "8px 0 0" },
  row: { display: "flex", justifyContent: "space-between", padding: "3px 0" },
  dt: { color: "#8b93a0" },
  dd: { margin: 0, color: ACCENT, fontVariantNumeric: "tabular-nums" },
  canvas: { position: "relative", overflow: "hidden" },
  status: { background: "#101215", borderTop: "1px solid #2c313a", color: "#8b93a0",
            padding: "7px 14px" },
};
