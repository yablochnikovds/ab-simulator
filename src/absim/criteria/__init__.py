"""Statistical criteria for A/B testing.

Each criterion is a small dataclass exposing a single ``test(treatment,
control, **kwargs)`` method that returns a :class:`absim.TestResult`.
Importing this package side-effect-registers every concrete criterion in
:data:`absim.criteria.base.REGISTRY` — call :func:`available` to enumerate.
"""

from __future__ import annotations

from absim.criteria.base import Criterion, available, get, register
from absim.criteria.bootstrap import Bootstrap
from absim.criteria.cupac import CUPAC
from absim.criteria.cuped import CUPED
from absim.criteria.delta import DeltaMethod
from absim.criteria.linearization import Linearization
from absim.criteria.paired import PairedStratification
from absim.criteria.poststrat import PostStratification
from absim.criteria.welch import WelchTTest
from absim.criteria.zproportion import ZTestProportions

__all__ = [
    "CUPAC",
    "CUPED",
    "Bootstrap",
    "Criterion",
    "DeltaMethod",
    "Linearization",
    "PairedStratification",
    "PostStratification",
    "WelchTTest",
    "ZTestProportions",
    "available",
    "get",
    "register",
]
