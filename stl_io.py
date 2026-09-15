"""
stl_io.py — binary STL reader, extracted verbatim from the original
api_core.py so the version that shipped with a passing test is the version
that runs. Pure NumPy: no trimesh, no meshio.

Kept in its own module rather than pasted into the rewritten api_core.py
because test_stl_parser.py reaches for it by source-slicing a file, and a
parser that lives in one place cannot drift from the one under test.
"""

import numpy as np


def parse_stl_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Binary STL reader with vertex welding.

    Welding is not optional here. STL stores every triangle independently
    with no shared topology, and the segmentation is a graph search over
    mesh edges -- on unwelded soup, every triangle is its own island and the
    region grow selects exactly one face. Exact bit-pattern matching is used
    rather than a tolerance so no vertex coordinate is altered.
    """
    if len(data) < 84:
        raise ValueError("File too short to be a binary STL.")
    n_tri = int(np.frombuffer(data[80:84], dtype="<u4")[0])
    expected = 84 + n_tri * 50
    if len(data) < expected:
        raise ValueError(
            f"Truncated file or ASCII STL: header declares {n_tri} triangles "
            f"({expected} bytes) but the file is {len(data)} bytes.")

    rec = np.frombuffer(data[84:expected], dtype=np.dtype([
        ("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")]))
    tri = rec["v"].reshape(-1, 3).astype(np.float64)
    uniq, inverse = np.unique(tri, axis=0, return_inverse=True)
    return uniq, np.asarray(inverse).reshape(-1, 3).astype(np.int64)


# alias so api_core.py's `cg.load_stl_bytes(...)` call site reads naturally
load_stl_bytes = parse_stl_bytes
