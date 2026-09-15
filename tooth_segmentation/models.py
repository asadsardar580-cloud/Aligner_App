"""Data models. Candidates are NOT given FDI numbers - that is a later stage."""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class PreprocessReport:
    n_vertices: int
    n_faces: int
    n_connected_components: int
    n_boundary_edges: int
    n_nonmanifold_edges: int
    n_degenerate_faces: int
    n_duplicate_vertices_merged: int
    watertight: bool
    bounding_box: tuple
    timings: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"{self.n_vertices:,} verts / {self.n_faces:,} faces | "
                f"components={self.n_connected_components} | "
                f"boundary edges={self.n_boundary_edges:,} | "
                f"non-manifold={self.n_nonmanifold_edges:,} | "
                f"watertight={self.watertight}")


@dataclass
class ToothCandidate:
    """Identified as candidate_NNN only. FDI numbering comes later."""
    id: str
    face_mask: np.ndarray
    vertex_count: int = 0
    triangle_count: int = 0
    surface_area: float = 0.0
    centroid: Optional[np.ndarray] = None
    bounding_box: Optional[tuple] = None
    principal_axes: Optional[np.ndarray] = None
    height: float = 0.0
    width: float = 0.0
    depth: float = 0.0
    curvature_mean: float = 0.0
    curvature_std: float = 0.0
    neighbours: list = field(default_factory=list)
    confidence: float = 0.0
    warnings: list = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return self.confidence < 0.70 or bool(self.warnings)


@dataclass
class ArchFrame:
    """Initial geometric reference from PCA ONLY - not anatomical tooth axes."""
    centroid: np.ndarray
    occlusal_normal: np.ndarray
    arch_width_axis: np.ndarray
    arch_depth_axis: np.ndarray
    explained_variance: np.ndarray
