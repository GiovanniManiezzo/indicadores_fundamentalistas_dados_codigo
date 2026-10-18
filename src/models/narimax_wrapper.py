"""NARIMAX: Nonlinear ARIMAX = NARX aplicado à série diferenciada (d=1).

Herda NARXWrapper com mode="diff" por padrão.
"""
from __future__ import annotations
from .narx_wrapper import NARXWrapper


class NARIMAXWrapper(NARXWrapper):
    """NARIMAX: d=1 via mode='diff'; NARX sobre ∇y."""

    def __init__(
        self,
        p: int = 3,
        degree: int = 2,
        alpha: float = 1.0,
        mode: str = "diff",
        boxcox_lambda: float | None = None,
    ) -> None:
        super().__init__(p=p, degree=degree, alpha=alpha, mode=mode, boxcox_lambda=boxcox_lambda)

    def get_params_str(self) -> str:
        return f"NARIMAX(p={self.p},deg={self.degree})"
