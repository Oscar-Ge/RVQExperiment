"""
RFSQ Model Definitions for RVQ Experiment.

This package contains:
- ActionRFSQAE: RFSQ AutoEncoder (Phase 1)
- RFSQDraftModelWithProjection: Draft Model (Phase 2 - predicts L0-L2)
- ConditionedRFSQHead: Conditional Main Model (Phase 2 - predicts all layers)
"""

from .rfsq_models import (
    ActionRFSQAE,
    RFSQDraftModelWithProjection,
    ConditionedRFSQHead,
    RobustSTEQuantizer,
    RobustRFSQBlock
)

__all__ = [
    'ActionRFSQAE',
    'RFSQDraftModelWithProjection',
    'ConditionedRFSQHead',
    'RobustSTEQuantizer',
    'RobustRFSQBlock'
]
