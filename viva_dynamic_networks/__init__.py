"""viva_dynamic_networks — dynamic-network dynamics as process-bigraph reactions.

Re-exports the processes and the visualization, and imports the composites
subpackage for its ``@composite_generator`` side effects so the dashboard
discovers every rule.
"""
from .processes import NetworkDynamicsProcess, NetworkMetrics
from .sbml_nodes import NetworkSBMLReactors, NetworkTransport, SpeciesSyncMetric
from .viz import NetworkEvolution

from . import composites  # noqa: F401  (registers composite generators)

__all__ = [
    "NetworkDynamicsProcess",
    "NetworkMetrics",
    "NetworkSBMLReactors",
    "NetworkTransport",
    "SpeciesSyncMetric",
    "NetworkEvolution",
]
