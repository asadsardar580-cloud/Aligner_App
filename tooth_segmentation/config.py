"""Configuration. Every weight is configurable here, not hard-coded at call sites."""
from dataclasses import dataclass, field


@dataclass
class BoundaryWeights:
    """Relative contribution of each feature to boundary_score in [0,1]."""
    curvature_weight: float = 1.0
    normal_weight: float = 0.6
    concavity_weight: float = 1.4
    valley_weight: float = 1.0
    orientation_weight: float = 0.4


@dataclass
class SegmentationConfig:
    boundary: BoundaryWeights = field(default_factory=BoundaryWeights)

    smoothing_iterations: int = 2
    barrier_weight: float = 25.0

    min_candidate_fraction: float = 0.005
    max_candidate_fraction: float = 0.35
    min_candidate_faces: int = 50

    confidence_review_threshold: float = 0.70
    deterministic: bool = True
