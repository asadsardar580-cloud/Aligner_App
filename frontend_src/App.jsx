import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const API = "http://127.0.0.1:8000";

const GINGIVA = new THREE.Color("#d9a7a0");
const SELECTED = new THREE.Color("#3fa9ff");
const PALETTE = ["#f2ede0", "#e8e0c8", "#f5efe2", "#ded6bd", "#efe7d4",
                 "#e3dcc4", "#f7f1e4", "#d9d1b8"];

// Mapped FDI numbers to colors as Claude requested
const FDI_COLOURS = {};
[11,12,13,14,15,16,17,18, 21,22,23,24,25,26,27,28, 31,32,33,34,35,36,37,38, 41,42,43,44,45,46,47,48].forEach((fdi, i) => {
    FDI_COLOURS[fdi] = new THREE.Color(PALETTE[i % PALETTE.length]).toArray();
});

export default function App() {
  const mountRef = useRef(null);
  const three = useRef({});
  const pointerDown = useRef(null);
  const arches = useRef({});          

  const [status, setStatus] = useState("Load an arch to begin.");
  const [sessions, setSessions] = useState({});
  const [active, setActive] = useState("maxillary");
  const [busy, setBusy] = useState(false);

  // ---------------------------------------------------------------- scene
  useEffect(() => {
    const mount = mountRef.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d0f12);

    const camera = new THREE.PerspectiveCamera(45, mount.clientWidth / mount.clientHeight, 0.1, 5000);
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

    return () => {
      cancelAnimationFrame(raf);
      controls.dispose(); renderer.dispose();
      if (renderer.domElement.parentNode) mount.removeChild(renderer.domElement);
    };
  }, []);

  // --- CLAUDE'S EXACT ALIGNMENT LOGIC ---
  
  const applyLabels = (geom, labels) => {
    const n = geom.attributes.position.count;
    if (labels.length !== n)              // fail loudly rather than mis-colour
      throw new Error(`labels ${labels.length} vs vertices ${n} — wrong mesh`);
    const c = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const [r,g,b] = FDI_COLOURS[labels[i]] ?? [0.85,0.75,0.72];   // gingiva
      c[i*3]=r; c[i*3+1]=g; c[i*3+2]=b;
    }
    geom.setAttribute("color", new THREE.BufferAttribute(c, 3));
  };

  const loadArch = useCallback(async (event, archName) => {
    const fileObj = event.target.files?.[0];
    if (!fileObj) return;
    setBusy(true);
    setStatus(`Uploading and Conditioning ${fileObj.name}...`);

    try {
      const fd = new FormData();
      fd.append("arch", archName === "maxillary" ? "upper" : "lower");
      fd.append("file", fileObj);
      
      const resSession = await fetch(`${API}/api/session`, {method:"POST", body:fd});
      if (!resSession.ok) throw new Error(`${resSession.status} ${await resSession.text()}`);
      const s = await resSession.json();

      const resMesh = await fetch(`${API}/api/session/${s.session_id}/mesh`);
      const m = await resMesh.json();
      
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(m.positions), 3));
      geom.setIndex(new THREE.BufferAttribute(new Uint32Array(m.indices), 1));
      geom.computeVertexNormals();

      const { scene, camera, controls } = three.current;
      if (arches.current[archName]) {
        scene.remove(arches.current[archName].mesh);
        arches.current[archName].mesh.geometry.dispose();
      }

      // Default color before segmentation
      const nVerts = geom.attributes.position.count;
      const colors = new Float32Array(nVerts * 3);
      for (let i = 0; i < nVerts; i++) GINGIVA.toArray(colors, i * 3);
      geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));

      const mesh = new THREE.Mesh(geom, new THREE.MeshPhongMaterial({
        vertexColors: true, specular: 0x222222, shininess: 22, side: THREE.DoubleSide,
      }));
      scene.add(mesh);
      arches.current[archName] = { mesh, geometry: geom };

      // DO NOT call geom.center(), geom.scale(), or translate the mesh.
      const bs = geom.boundingSphere ?? (geom.computeBoundingSphere(), geom.boundingSphere);
      camera.position.set(bs.center.x, bs.center.y - bs.radius*2.5, bs.center.z + bs.radius*1.5);
      controls.target.copy(bs.center);

      setSessions((prev) => ({ ...prev, [archName]: s }));
      setActive(archName);
      setStatus(`Arch Loaded: ${s.face_count.toLocaleString()} faces conditioned. Click "Segment Teeth".`);
    } catch (err) {
      setStatus(`Failed: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }, []);

  const runSegmentation = useCallback(async () => {
    const sid = sessions[active]?.session_id;
    if (!sid) return;
    setBusy(true); setStatus("AI Segmenting teeth...");
    try {
      const res = await fetch(`${API}/api/session/${sid}/segment`, { method: "POST" });
      if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
      const data = await res.json();
      
      applyLabels(arches.current[active].geometry, data.labels);
      setStatus(`Segmentation Complete! Validated ${data.jaw} FDI Numbers.`);
    } catch (err) {
      setStatus(`Segmentation failed: ${err.message}`);
    } finally { setBusy(false); }
  }, [sessions, active]);

  return (
    <div style={S.app}>
      <aside style={S.rail}>
        <h1 style={S.title}>Clinical Micro-Planner</h1>
        <section style={S.group}>
          <div style={S.legend}>ARCHES</div>
          {["maxillary", "mandibular"].map((a) => (
            <div key={a} style={{ marginBottom: 8 }}>
              <label style={S.button}>
                Load {a} STL
                <input type="file" accept=".stl,.obj" disabled={busy} onChange={(e) => loadArch(e, a)} style={{ display: "none" }} />
              </label>
            </div>
          ))}
        </section>
        
        <section style={S.group}>
          <div style={S.legend}>AI CLASSIFICATION</div>
          <button onClick={runSegmentation} disabled={busy || !sessions[active]} style={S.primary}>
            Segment Teeth
          </button>
        </section>
      </aside>
      <main ref={mountRef} style={S.canvas} />
      <footer style={S.status}>{status}</footer>
    </div>
  );
}

const S = {
  app: { display: "grid", gridTemplateColumns: "300px 1fr", gridTemplateRows: "1fr auto", height: "100vh", background: "#0d0f12", color: "#e7ebee", fontFamily: "system-ui, sans-serif", fontSize: 13 },
  rail: { gridRow: "1 / 3", background: "#1a1d22", borderRight: "1px solid #2c313a", padding: 16, overflowY: "auto" },
  title: { fontSize: 15, fontWeight: 600, margin: "0 0 18px" },
  group: { border: "1px solid #2c313a", borderRadius: 10, padding: 12, marginBottom: 12, background: "rgba(255,255,255,0.02)" },
  legend: { color: "#3fc6d4", fontSize: 10, fontWeight: 700, letterSpacing: 1.2, marginBottom: 10 },
  button: { display: "block", background: "linear-gradient(#2a2f37,#21252b)", border: "1px solid #2c313a", borderRadius: 6, padding: "8px 12px", cursor: "pointer", textAlign: "center" },
  primary: { width: "100%", background: "linear-gradient(#3fc6d4,#2a8b96)", color: "#06181a", border: "none", borderRadius: 6, padding: "8px 12px", fontWeight: 600, cursor: "pointer" },
  canvas: { position: "relative", overflow: "hidden" },
  status: { background: "#101215", borderTop: "1px solid #2c313a", color: "#8b93a0", padding: "7px 14px" },
};