"""How much interproximal signal survives realistic contact blending?"""
import numpy as np, core_geometry as cg
from arch_fixture import synthetic_arch

CROWN_R = 3.2
def measure(spacing, blend):
    verts, faces, centres = synthetic_arch(n_teeth=4, spacing=spacing,
                                           crown_r=CROWN_R, grid=150, blend=blend)
    edges = cg.directed_edges(faces)
    conc = cg.smooth_scalar_fast(cg.vertex_concavity_fast(verts, faces, edges=edges),
                                 faces, 2, edges=edges)
    mid = (centres[1] + centres[2]) / 2.0
    interprox = conc[np.linalg.norm(verts[:, :2] - mid, axis=1) < 0.8].mean()
    d = np.linalg.norm(verts[:, :2] - centres[0], axis=1)
    sulcus = conc[(d > CROWN_R-0.4) & (d < CROWN_R+0.4)].mean()
    return sulcus, interprox, (interprox/sulcus if sulcus > 1e-9 else 0)

print("Interproximal signal strength, relative to the gingival sulcus (1.0 = as strong)\n")
print(f"{'contact':>22} |" + "".join(f"{f'{s}mm':>9}" for s in [8.0, 7.0, 6.4, 6.0, 5.5]))
print("-"*74)
for label, blend in [("sharp crease (hard max)", None), ("lightly blended", 6.0),
                     ("realistic contact", 3.0), ("heavily merged", 1.5)]:
    row = f"{label:>22} |"
    for sp in [8.0, 7.0, 6.4, 6.0, 5.5]:
        _, _, ratio = measure(sp, blend)
        row += f"{ratio:9.2f}"
    print(row)

print("\nReading: values near 1.0 mean the tooth/tooth boundary is as detectable as")
print("the gumline. Values near 0 mean there is no geometric boundary to find.")
