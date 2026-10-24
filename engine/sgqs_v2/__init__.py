"""Training-only SGQS v2 candidate allocation; v1 remains reproducible."""
from .candidate_supervision import SGQSV2Supervisor
from .criterion import SGQSV2Criterion

__all__ = ['SGQSV2Supervisor', 'SGQSV2Criterion']
