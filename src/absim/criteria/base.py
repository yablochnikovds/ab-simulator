"""Criterion protocol and registry.

A criterion is anything that can take a sample and return a :class:`TestResult`.
The :data:`REGISTRY` lookup keeps the simulator + CLI decoupled from concrete
implementations, so adding a new criterion is a single-file change (the new
module just calls :func:`register`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from absim.types import FloatArray, TestResult


@runtime_checkable
class Criterion(Protocol):
    """Protocol every statistical criterion must implement.

    Concrete criteria are typically dataclasses configured with their
    hyperparameters (e.g. ``alpha``, ``B`` for bootstrap), and the actual
    work is done in :meth:`test`.
    """

    name: str

    def test(
        self,
        treatment: FloatArray,
        control: FloatArray,
        **kwargs: Any,
    ) -> TestResult:
        """Run the test and return its :class:`TestResult`.

        ``**kwargs`` lets callers pass auxiliary data needed by some criteria
        (covariates for CUPED, denominators for ratio metrics, strata, ...).
        Criteria must ignore unknown keys.
        """
        ...


REGISTRY: dict[str, type[Any]] = {}

_C = TypeVar("_C", bound=type[Any])


def register(name: str) -> Callable[[_C], _C]:
    """Class decorator: register a criterion class under ``name``.

    Used as ``@register("welch_t")`` above the class definition. Decoupling
    the registry key from a class attribute lets concrete criteria use
    ``slots=True`` (which would otherwise strip defaults from the class dict).
    """

    def _decorator(cls: _C) -> _C:
        if name in REGISTRY:
            raise ValueError(f"criterion {name!r} is already registered")
        REGISTRY[name] = cls
        return cls

    return _decorator


def get(name: str) -> type[Any]:
    """Return the criterion class registered under ``name``."""
    try:
        return REGISTRY[name]
    except KeyError as exc:
        available_names = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown criterion {name!r}. Available: {available_names}") from exc


def available() -> list[str]:
    """Return the sorted list of registered criterion names."""
    return sorted(REGISTRY)
