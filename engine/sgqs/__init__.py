"""Scale-aware GT Query Survival (SGQS) training components."""

from .candidate_supervision import SGQSQuerySupervisor, assign_scale_aware_owners
from .criterion import SGQSCriterion

__all__ = [
    'SGQSQuerySupervisor',
    'SGQSCriterion',
    'assign_scale_aware_owners',
]
