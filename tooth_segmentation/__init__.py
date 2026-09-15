"""tooth_segmentation — modular automatic tooth segmentation.

CLINICAL NOTE (spec section 20): every result is computationally generated
and requires clinician verification. No output claims clinical accuracy.
"""
from .config import SegmentationConfig, BoundaryWeights
from .models import PreprocessReport, ToothCandidate, ArchFrame
from . import mesh_preprocessor, normals, curvature, arch_geometry, label_adapter

__all__ = ["SegmentationConfig", "BoundaryWeights", "PreprocessReport",
           "ToothCandidate", "ArchFrame", "mesh_preprocessor", "normals",
           "curvature", "arch_geometry", "label_adapter"]
