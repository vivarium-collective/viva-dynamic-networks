"""Composite generators for the dynamic-networks investigation.

Each generator is a ``@composite_generator`` that the dashboard's Composites tab
discovers: one per Family-A dynamic rule. They all build the same shape — the
generalized ``NetworkDynamicsProcess`` driving a seed graph, a ``NetworkMetrics``
process emitting scalar observables, a RAMEmitter (for the live network
animation) and a declared ParquetEmitter (so study ``behavior_tests`` evaluate).
"""
from . import family_a  # noqa: F401
from . import family_b  # noqa: F401
from . import family_c  # noqa: F401
from .family_a import (
    activity_driven,
    adaptive_rewire,
    duplication_divergence,
    erdos_renyi,
    fitness_attachment,
    forest_fire,
    preferential_attachment,
    small_world_rewire,
)
from .family_b import (
    repressilator_network_coupled,
    repressilator_network_uncoupled,
)
from .family_c import (
    growing_repressilator_network,
    rewiring_repressilator_network,
)

__all__ = [
    "erdos_renyi",
    "preferential_attachment",
    "fitness_attachment",
    "duplication_divergence",
    "forest_fire",
    "small_world_rewire",
    "adaptive_rewire",
    "activity_driven",
    "repressilator_network_coupled",
    "repressilator_network_uncoupled",
    "growing_repressilator_network",
    "rewiring_repressilator_network",
]
