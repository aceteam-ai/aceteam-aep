"""Shared pytest configuration."""

from __future__ import annotations

import os

# The safety detectors load HuggingFace models that ship only `pytorch_model.bin`.
# transformers reacts by spawning a *non-daemon* "Thread-auto_conversion" thread that
# calls out to the safetensors conversion Space over the network. That thread outlives
# the tests, so the interpreter blocks on it at shutdown and the run hangs after the
# summary line is printed. transformers exposes this env var precisely to opt out.
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
