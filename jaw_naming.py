"""
jaw_naming.py — get the jaw label right, and prove it afterwards.
"""

import os
import re

UPPER_FDI = set(range(11, 19)) | set(range(21, 29))   # 11-18, 21-28
LOWER_FDI = set(range(31, 39)) | set(range(41, 49))   # 31-38, 41-48

def inspect_get_jaw(tgn_root="ToothGroupNetwork"):
    path = os.path.join(tgn_root, "predict_utils.py")
    if not os.path.exists(path):
        return f"not found: {path}"
    src = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"[ \t]*def get_jaw\b.*?(?=\n[ \t]*def |\n[ \t]*class |\Z)", src, re.S)
    if not m:
        return "get_jaw not found in predict_utils.py"
    return m.group(0)

def scan_filename(jaw, patient_id="case", suffix=".obj"):
    jaw = str(jaw).strip().lower()
    if jaw not in ("upper", "lower"):
        raise ValueError(f"jaw must be 'upper' or 'lower', got {jaw!r}.")
    pid = re.sub(r"[^A-Za-z0-9]", "", str(patient_id)) or "case"
    return f"{pid}_{jaw}{suffix}"

def jaw_for_arch(arch):
    a = str(arch).strip().lower()
    if a in ("maxilla", "maxillary", "upper", "u"):
        return "upper"
    if a in ("mandible", "mandibular", "lower", "l"):
        return "lower"
    raise ValueError(f"unrecognised arch: {arch!r}")

def extract_labels(source, expect_jaw=None):
    meta = {"source_shape": None, "unparseable": [], "vertex_count": None, "jaw_in_file": None}

    if isinstance(source, dict):
        if "labels" in source:
            meta["source_shape"] = "challenge_json"
            meta["jaw_in_file"] = source.get("jaw")
            raw = source["labels"]
        else:
            meta["source_shape"] = "dict_keyed_by_tooth"
            raw = list(source.keys())
    elif isinstance(source, (list, tuple)):
        meta["source_shape"] = "array"
        raw = list(source)
    else:
        raise TypeError("cannot extract labels from source")

    labels = []
    for v in raw:
        try:
            if isinstance(v, bool):
                raise ValueError("bool is not a label")
            labels.append(int(v))
        except (TypeError, ValueError):
            meta["unparseable"].append(v)

    meta["vertex_count"] = len(labels)
    if expect_jaw and meta["jaw_in_file"]:
        meta["jaw_matches_file"] = (str(meta["jaw_in_file"]).strip().lower() == str(expect_jaw).strip().lower())
    return labels, meta

def verify_fdi_matches_jaw(labels, jaw, ignore=(0,), min_vertices_per_tooth=20):
    jaw = str(jaw).strip().lower()
    values, meta = extract_labels(labels, expect_jaw=jaw)

    expected = UPPER_FDI if jaw == "upper" else LOWER_FDI
    other = LOWER_FDI if jaw == "upper" else UPPER_FDI

    counts = {}
    for v in values:
        if v in ignore:
            continue
        counts[v] = counts.get(v, 0) + 1

    present = sorted(counts)
    in_expected = [v for v in present if v in expected]
    in_other = [v for v in present if v in other]
    unknown = [v for v in present if v not in expected and v not in other]
    slivers = {v: n for v, n in counts.items() if n < min_vertices_per_tooth and v in expected}

    result = {
        "jaw_declared": jaw,
        "source_shape": meta["source_shape"],
        "vertex_count": meta["vertex_count"],
        "jaw_recorded_in_file": meta["jaw_in_file"],
        "labels_present": present,
        "vertices_per_tooth": counts,
        "matching_declared_jaw": in_expected,
        "belonging_to_opposite_jaw": in_other,
        "outside_fdi_range": unknown,
        "unparseable_entries": meta["unparseable"],
        "sliver_regions": slivers,
    }
    result["ok"] = bool(in_expected) and not in_other and not unknown and not meta["unparseable"] and not slivers
    
    if "jaw_matches_file" in meta:
        result["jaw_matches_file"] = meta["jaw_matches_file"]
        if not meta["jaw_matches_file"]:
            result["ok"] = False

    if meta["unparseable"]:
        result["diagnosis"] = f"Could not interpret {len(meta['unparseable'])} entries."
    elif result.get("jaw_matches_file") is False:
        result["diagnosis"] = f"The output file records jaw={meta['jaw_in_file']!r} but this scan was submitted as {jaw!r}."
    elif in_other:
        result["diagnosis"] = f"Labels {in_other} belong to the wrong arch."
    elif unknown:
        result["diagnosis"] = f"Labels {unknown} are not valid FDI numbers."
    elif not in_expected:
        result["diagnosis"] = "No tooth labels at all — segmentation returned only background."
    elif slivers:
        result["diagnosis"] = f"Teeth {sorted(slivers)} have fewer than {min_vertices_per_tooth} vertices each."
    else:
        result["diagnosis"] = "Labels are consistent with the declared arch."
    return result

def drop_third_molars(labels, replace_with=0):
    thirds = {18, 28, 38, 48}
    return [replace_with if int(v) in thirds else v for v in labels]