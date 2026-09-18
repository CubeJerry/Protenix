"""Opt-in inference reuse and checkpoint-only linear initialization."""

import os
from contextlib import contextmanager
from contextvars import ContextVar

from torch import nn

_SKIP_RANDOM_INIT = ContextVar("protenix_skip_random_init", default=False)


def enabled(name: str) -> bool:
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def skipping_random_init() -> bool:
    return _SKIP_RANDOM_INIT.get()


@contextmanager
def checkpoint_initialization(skip: bool, strict: bool):
    """Scope to model construction followed immediately by a strict load.

    Does not patch torch globally or change buffer/constant initialization.
    Callers must reseed before stochastic inference, as infer_predict does.
    """
    if skip and not strict:
        raise ValueError(
            "Skipping random initialization requires a strict checkpoint load"
        )
    token = _SKIP_RANDOM_INIT.set(skip)
    try:
        yield
    finally:
        _SKIP_RANDOM_INIT.reset(token)


class CheckpointLinear(nn.Linear):
    """Same parameters/forward as nn.Linear; optional checkpoint-only construction."""

    def reset_parameters(self) -> None:
        if not skipping_random_init():
            super().reset_parameters()
