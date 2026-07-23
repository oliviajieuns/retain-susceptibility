"""Fail-closed evidence contracts for the paper-facing result pipeline.

The modules in this package deliberately sit downstream of experiment
runners.  They turn a predeclared setting/parent roster and a normalized
evidence ledger into explicit row decisions, table-readiness diagnostics, and
LaTeX status macros.  Missing work is retained in every denominator and can
never be interpreted as a successful claim.
"""

from .decisions import evaluate_evidence
from .registry import EvidenceContract, load_contract
from .schemas import EvidenceLedger, EvidenceValidationError

__all__ = [
    "EvidenceContract",
    "EvidenceLedger",
    "EvidenceValidationError",
    "evaluate_evidence",
    "load_contract",
]
