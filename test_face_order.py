"""Does the server's face index i correspond to triangle i in the STL file?

If not, every face index the server returns highlights the WRONG triangle in
Three.js. THREE.STLLoader emits non-indexed geometry in file order, so this has
to hold for the browser and server to agree — it underpins /wand's vertex_ids
and /cut's removed_faces alike.

The parser moved out of api_core into stl_io (api_core imports it at :20), so
this test called a name that had not existed for some time. It is the parser
that is under test, not the web layer, so it imports stl_io directly and no
longer drags in FastAPI, torch and the whole geometry engine to check a struct.
"""
import io, struct
import numpy as np
import stl_io

# distinctive triangles so order is unambiguous
tris = np.array([
    [[0,0,0],[1,0,0],[0,1,0]],
    [[5,5,5],[6,5,5],[5,6,5]],
    [[9,0,0],[9,1,0],[9,0,1]],
    [[0,0,0],[1,0,0],[0,0,1]],      # shares corners with tri 0 -> forces welding
    [[2,2,2],[3,2,2],[2,3,2]],
], dtype=np.float32)

buf = io.BytesIO(); buf.write(b'\0'*80); buf.write(struct.pack('<I', len(tris)))
for t in tris:
    buf.write(struct.pack('<3f',0,0,0))
    for v in t: buf.write(struct.pack('<3f', *v))
    buf.write(struct.pack('<H',0))
stl = buf.getvalue()

verts, faces = stl_io.parse_stl_bytes(stl)
print(f"{len(tris)} triangles in file -> {len(verts)} welded verts, {len(faces)} faces")
assert len(faces) == len(tris), "face count changed - order cannot be preserved"

for i, t in enumerate(tris):
    server_tri = verts[faces[i]]
    # same three corners, allowing for welding not reordering corners
    assert np.allclose(np.sort(server_tri, axis=0), np.sort(t.astype(float), axis=0)), \
        f"face {i} does not match file triangle {i}"
print("PASS  server face i == file triangle i, for every face")
print("      -> server face indices index THREE.STLLoader geometry directly")

# and welding actually happened (shared corners collapsed)
assert len(verts) < len(tris)*3, "welding did not occur"
print(f"PASS  welding collapsed {len(tris)*3} loose corners to {len(verts)} shared vertices")
