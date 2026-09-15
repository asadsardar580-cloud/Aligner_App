import numpy as np, struct, io, sys
sys.modules['fastapi'] = type(sys)('fastapi')  # stub so we can import the parser alone
import importlib.util
spec = importlib.util.spec_from_file_location("srv_parse", "server.py")

# extract just parse_stl_bytes without importing fastapi
src = open('server.py').read()
start = src.index('def parse_stl_bytes'); end = src.index('def _expire_sessions')
ns = {'np': np}
exec(src[start:end], ns)
parse_stl_bytes = ns['parse_stl_bytes']

def write_binary_stl(verts, faces):
    buf = io.BytesIO()
    buf.write(b'\0'*80); buf.write(struct.pack('<I', len(faces)))
    for f in faces:
        buf.write(struct.pack('<3f', 0,0,0))
        for vi in f: buf.write(struct.pack('<3f', *verts[vi]))
        buf.write(struct.pack('<H', 0))
    return buf.getvalue()

# tetrahedron: 4 shared verts, 4 faces. STL stores 12 loose verts.
v = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
f = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
data = write_binary_stl(v, f)
pv_, pf_ = parse_stl_bytes(data)
assert len(pv_) == 4, f"welding failed: got {len(pv_)} verts, expected 4 (STL had 12 loose)"
assert len(pf_) == 4, f"face count wrong: {len(pf_)}"
import core_geometry as cg
assert cg.is_edge_manifold_closed(pf_), "welded tetrahedron should be closed/manifold"
print(f"PASS  STL parser: 12 loose STL verts welded to {len(pv_)} shared, "
      f"{len(pf_)} faces, topology closed (edge graph usable by Dijkstra)")

try:
    parse_stl_bytes(b'x'*40); print("FAIL  should reject short file")
except ValueError as e: print(f"PASS  rejects truncated file: {str(e)[:40]}...")
try:
    bad = b'\0'*80 + struct.pack('<I', 99999) + b'\0'*100
    parse_stl_bytes(bad); print("FAIL  should reject truncated body")
except ValueError as e: print(f"PASS  rejects header/body mismatch (catches ASCII STL too)")
