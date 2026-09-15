import numpy as np, json, io, time, gzip, base64

# Representative full-arch intraoral scan
n_verts, n_faces = 360_000, 718_000
verts = np.random.rand(n_verts,3).astype(np.float32)*40
faces = np.random.randint(0,n_verts,(n_faces,3)).astype(np.int32)

def mb(b): return b/1_048_576

# --- binary STL on the wire (what the client would upload) ---
stl_bytes = 84 + n_faces*50
print(f"Full-arch scan: {n_verts:,} verts / {n_faces:,} faces")
print(f"  Binary STL upload size            : {mb(stl_bytes):7.1f} MB")

# --- how the meshes come BACK (crown + base) ---
t=time.time()
payload = {"verts": verts.tolist(), "faces": faces.tolist()}
js = json.dumps(payload).encode()
t_json = time.time()-t
print(f"  JSON (verts+faces as nested lists) : {mb(len(js)):7.1f} MB   encode {t_json:5.2f}s")

t=time.time()
buf = io.BytesIO(); np.savez(buf, verts=verts, faces=faces); raw = buf.getvalue()
t_npz = time.time()-t
print(f"  .npz uncompressed binary           : {mb(len(raw)):7.1f} MB   encode {t_npz:5.2f}s")

t=time.time()
buf = io.BytesIO(); np.savez_compressed(buf, verts=verts, faces=faces); comp = buf.getvalue()
t_npzc = time.time()-t
print(f"  .npz compressed                    : {mb(len(comp)):7.1f} MB   encode {t_npzc:5.2f}s")

print(f"\n  JSON is {len(js)/len(raw):.1f}x larger than raw binary and {t_json/t_npz:.0f}x slower to encode.")

# --- wall-clock over realistic clinic links, vs local compute ---
print("\nRound-trip time to segment ONE tooth (upload scan + download crown+base):")
total_bytes = stl_bytes + len(raw)
for label, mbps in [("Clinic DSL   5 Mbps up", 5), ("Business    25 Mbps up", 25),
                    ("Fiber      100 Mbps up", 100)]:
    secs = (total_bytes*8)/(mbps*1_000_000)
    print(f"  {label}: {secs:6.1f}s network  vs  0.46s local compute  ->  {secs/0.46:5.0f}x slower")
