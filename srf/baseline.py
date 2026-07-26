"""baseline.py — no-op method module (vanilla model, no intervention).

Implements the same interface as srf.py / vaf.py so eval.py can dispatch
to it without modification. All methods are no-ops; the model runs unmodified.
"""
from __future__ import annotations


def setup(model, processor, calib_dataset: str | None = None) -> None:
    """No calibration needed for baseline."""
    pass


def reset_for_dataset(dataset: str | None = None, **kwargs) -> None:
    """No per-dataset state to reset."""
    pass


def prepare_sample(
    inp,
    img_start: int,
    img_end: int,
    image,
    question: str,
    model,
    processor,
    noun_override: str | None = None,
) -> None:
    """No hooks to install — model runs unmodified."""
    pass


def cleanup() -> None:
    """No hooks to remove."""
    pass
