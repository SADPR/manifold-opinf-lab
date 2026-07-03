"""Small shared utilities for manifold Operator Inference experiments."""

from .features import (
    compact_monomial_matrix,
    compact_monomials,
    continuous_feature_matrix,
    continuous_feature_vector,
    polynomial_parameter_features,
)
from .pod import energy_captured, project_snapshots, reconstruct_snapshots
from .regression import fit_operator
from .time_integration import rollout_rk4

__all__ = [
    "compact_monomial_matrix",
    "compact_monomials",
    "continuous_feature_matrix",
    "continuous_feature_vector",
    "energy_captured",
    "fit_operator",
    "polynomial_parameter_features",
    "project_snapshots",
    "reconstruct_snapshots",
    "rollout_rk4",
]
