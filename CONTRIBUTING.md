# Contributing to absim

## Dev environment

```bash
git clone https://github.com/yablochnikov/ab-simulator
cd ab-simulator
uv sync                       # installs all dev + docs deps
uv run pre-commit install     # set up the local hooks
```

## Running checks

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest                                      # unit + property
uv run pytest -m statistical                       # FPR / power validation (slower)
uv run pytest --cov=src/absim --cov-fail-under=85  # full coverage gate
uv run mkdocs build --strict                       # docs build
```

## Adding a criterion

A criterion is a `@dataclass(frozen=True, slots=True)` exposing a `test`
method that returns an `absim.TestResult`. The whole change is a single
new file:

```python
# src/absim/criteria/my_method.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np

from absim.criteria.base import register
from absim.types import TestResult


@register("my_method")
@dataclass(frozen=True, slots=True)
class MyMethod:
    """One-line summary; full reference in docs/criteria/my_method.md."""

    alpha: float = 0.05
    name: str = "my_method"

    def test(self, treatment, control, **kwargs: Any) -> TestResult:
        # ... your math here ...
        return TestResult(
            p_value=p, statistic=t, effect=delta,
            std_error=se, ci_low=lo, ci_high=hi,
            rejected=p < self.alpha,
        )
```

Then:

1. Re-export it from `src/absim/criteria/__init__.py`.
2. Add a `conf/criterion/my_method.yaml` with the `_target_:` for Hydra.
3. Add a unit test (textbook closed-form), a property test (a stat
   invariant), and — if the method is a real production candidate — a
   statistical FPR test in `tests/statistical/`.
4. Document the formula and assumptions in `docs/criteria/my_method.md`.

## Adding a generator

Same pattern in `src/absim/generators/`. Implement `sample(rng, mean_shift)
-> Sample`. Emit auxiliary arrays under the standard keys
(`covariate_*`, `strata_*`, `numerator_*`, `denominator_*`,
`features_*`) so existing criteria can consume them.

## Commit style

[Conventional Commits](https://www.conventionalcommits.org/). Keep
messages short, imperative, ≤72 char subject. Examples:

```text
feat(criteria): add Mann–Whitney U test
fix(simulator): correct seed split when n_jobs=1
docs(criteria): clarify CUPED variance-reduction formula
```

## Releasing

We use a single source of truth in `src/absim/_version.py`. Bump it,
update `CHANGELOG.md`, tag, push.
