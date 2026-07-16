"""GUI-independent orchestration shared by the GUI and CLI front-ends."""

from .pipeline import (
    ClassifyConfig,
    ClassifySummary,
    PipelineError,
    run_classification,
)

__all__ = [
    "ClassifyConfig",
    "ClassifySummary",
    "PipelineError",
    "run_classification",
]
