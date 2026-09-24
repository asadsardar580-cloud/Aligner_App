import sys, os, json, hashlib
import numpy as np
import scipy.sparse.csgraph
from scipy.spatial import cKDTree
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core_geometry as cg
import stl_io
import self_intersection as si
import arch_frame

def digest(v, f):
    return hashlib.sha256(v.tobytes() + f.tobytes()).hexdigest()

def classify_pair(p, n_scan, n_wall):
    a, b = p
    type_a = "scan" if a < n_scan else ("wall" if a < n_scan + n_wall else "floor")
    type_b = "scan" if b < n_scan else ("wall" if b < n_scan + n_wall else "floor")
    key = f"{type_a}-{type_b}"
    if type_b + "-" + type_a in ["scan-wall", "scan-floor", "wall-floor"]: 
        key = f"{type_b}-{type_a}"
    return key

def main():
    with open("case_lower.stl", "rb") as f: raw = f.read()
    verts, faces = stl_io.parse_stl_bytes(raw)
    verts, faces, _ = cg.sanitize_scan(verts, faces)
    verts, faces, _ = cg.condition_mesh(verts, faces)
    with open("case_lower.stl_output.json", 'r') as lf: 
        case = json.load(lf)
    labels = np.array(case.get('labels', []))
    
    centroid = verts.mean(axis=0)
    d = verts - centroid
    _, _, vt = np.linalg.svd(d[::37], full_matrices=False)
    occ_axis = vt[2]
    slab = verts[(d @ occ_axis) > np.percentile(d @ occ_axis, 90)]
    if (slab - centroid).shape[0] < 10: slab = verts
    s = slab - slab.mean(axis=0)
    _, _, sv_ = np.linalg.svd(s[::7], full_matrices=False)
    along = s @ sv_[0]
    across = s @ sv_[1]
    pts = [slab[np.argmin(along)], slab[np.argmax(along)], slab[np.argmax(np.abs(across)) if abs(across).max() > 0 else 0]]
    frame = arch_frame.fit_occlusal_frame(list(map(float, pts[0])), list(map(float, pts[1])), list(map(float, pts[2])), centroid.tolist())
    
    tv, tf, tinfo = cg.trim_to_arch(verts, faces, frame, margin_mm=cg.ARCH_TRIM_MARGIN_MM)
    cv, cf, base_info = cg.build_cast_base(tv, tf, frame, rim=tinfo["rim_loop"])
    
    blob = cg.write_binary_stl_bytes(cv, cf)
    rv, rf = stl_io.parse_stl_bytes(blob)
    
    print("Digest (Float32-rounded):", digest(rv, rf))
    
    rep32 = si.self_intersection_report(rv, rf)
    print("Total pairs:", rep32["intersecting_pairs"])
    print("By kind:", rep32.get("by_kind", {}))
    
    n_scan = len(tf)
    n_wall = len(tinfo["rim_loop"]) * 2
    
    pairs32 = si.candidate_pairs(rv, rf)
    hit32, kind32 = si._pairs_intersect(rv, rf, pairs32, si.DEFAULT_TOUCH_TOL_MM)
    
    intersecting = pairs32[hit32]
    intersecting_kinds = kind32[hit32]
    
    counts = {"scan-wall": 0, "scan-floor": 0, "wall-wall": 0, "wall-floor": 0, "floor-floor": 0, "scan-scan": 0}
    for p in intersecting:
        counts[classify_pair(p, n_scan, n_wall)] += 1
        
    print("Per-class counts:", counts)
    
    origin, e1, e2, _ = cg._arch_basis(frame)
    
    # Identify clusters of intersecting faces
    # Connected component of faces in the cast mesh graph
    # Only faces involved in an intersection
    involved_faces = np.unique(intersecting)
    
    # Graph of faces sharing an edge
    edges = np.vstack([
        cf[:, [0, 1]],
        cf[:, [1, 2]],
        cf[:, [2, 0]]
    ])
    edges = np.sort(edges, axis=1)
    
    from collections import defaultdict
    grouped = defaultdict(list)
    face_idx = np.repeat(np.arange(len(cf)), 3)
    for i, (v1, v2) in enumerate(edges):
        grouped[(v1, v2)].append(face_idx[i])
        
    G = nx.Graph()
    G.add_nodes_from(involved_faces)
    
    for face_list in grouped.values():
        involved = [f for f in face_list if f in involved_faces]
        if len(involved) >= 2:
            for i in range(len(involved)):
                for j in range(i+1, len(involved)):
                    G.add_edge(involved[i], involved[j])
                    
    clusters = list(nx.connected_components(G))
    
    v_to_fdi = np.zeros(len(verts), dtype=int)
    if len(labels) > 0: v_to_fdi = labels[:len(verts)]
    
    # Distance to teeth
    graph_tv = cg.build_edge_graph(tv, tf, np.zeros(len(tv)))
    tooth_idx = np.where(v_to_fdi[:len(tv)] > 0)[0]
    dist_to_tooth = scipy.sparse.csgraph.dijkstra(graph_tv, directed=False, indices=tooth_idx, min_only=True) if len(tooth_idx) > 0 else np.full(len(tv), float('inf'))
    
    # Hole fill fan
    # Vertex valence
    valence = np.bincount(tf.ravel(), minlength=len(tv))
    
    scan_boundary = set(tinfo["rim_loop"])
    
    print(f"Number of clusters: {len(clusters)}")
    for i, cl in enumerate(clusters):
        f_list = list(cl)
        f_type = [("scan" if f < n_scan else ("wall" if f < n_scan + n_wall else "floor")) for f in f_list]
        scan_f = [f for f in f_list if f < n_scan]
        
        # nearest tooth
        if scan_f:
            cents = np.mean(rv[rf[scan_f]], axis=1)
            ds, closests = cKDTree(verts).query(cents)
            fdis = v_to_fdi[closests]
            fdi = fdis[fdis > 0][0] if np.any(fdis > 0) else 0
            
            y_coords = (cents - origin) @ e2
            side = "lingual" if np.mean(y_coords) < 0 else "buccal"
            
            touches_scan_boundary = any(v in scan_boundary for f in scan_f for v in tf[f])
            
            has_fan = any(valence[v] > 20 for f in scan_f for v in tf[f])
            
            min_dist = np.min([dist_to_tooth[v] for f in scan_f for v in tf[f]])
            max_dist = np.max([dist_to_tooth[v] for f in scan_f for v in tf[f]])
        else:
            fdi, side, touches_scan_boundary, has_fan, min_dist, max_dist = 0, "unknown", False, False, 0.0, 0.0
            
        print(f"Cluster {i+1}: {len(f_list)} faces. Nearest tooth: {fdi}, side: {side}, touches boundary: {touches_scan_boundary}, hole-fill fan: {has_fan}, dist to tooth: {min_dist:.2f} - {max_dist:.2f} mm")

    print("\nProjected rim self-crosses:")
    rim_pts = cv[tinfo["rim_loop"]]
    rim_pts2d = np.column_stack([(rim_pts - origin) @ e1, (rim_pts - origin) @ e2])
    def ccw(A,B,C): return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    def intersect(A,B,C,D): return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    
    crossings = []
    for i in range(len(rim_pts2d)):
        for j in range(i+2, len(rim_pts2d)):
            if i == 0 and j == len(rim_pts2d) - 1: continue
            if intersect(rim_pts2d[i], rim_pts2d[(i+1)%len(rim_pts2d)], rim_pts2d[j], rim_pts2d[(j+1)%len(rim_pts2d)]):
                # find closest tooth
                pt = (rim_pts2d[i] + rim_pts2d[j]) / 2
                pt3d = origin + pt[0] * e1 + pt[1] * e2
                d, cl = cKDTree(verts).query(pt3d)
                fdi = v_to_fdi[cl]
                crossings.append((i, j, pt, fdi))
    
    print(f"Total projected 2D rim crossings: {len(crossings)}")
    for i, j, pt, fdi in crossings[:5]:
        print(f"Segment {i} crosses {j} near tooth {fdi}")

if __name__ == "__main__":
    main()
