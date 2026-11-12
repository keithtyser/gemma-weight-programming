"""Harness templates and deterministic submission builders for E4B and 31B."""

from __future__ import annotations

from typing import Any

__all__ = ["HARNESS_PRESETS", "build_harness_package"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from v2.harnesses import builder

        return getattr(builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
