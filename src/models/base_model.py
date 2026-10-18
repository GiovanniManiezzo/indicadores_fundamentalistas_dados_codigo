"""Interface ABC comum para todos os wrappers de modelos."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseModel(ABC):
    @abstractmethod
    def fit(self, y: pd.Series, u: pd.DataFrame | None = None) -> "BaseModel":
        """Ajusta o modelo. y na escala original; u: exógenas ou None."""

    @abstractmethod
    def predict(self, steps: int = 1, u_future: pd.DataFrame | None = None) -> np.ndarray:
        """Retorna previsões na escala original (array de comprimento `steps`)."""

    @abstractmethod
    def get_params_str(self) -> str:
        """Descrição legível dos parâmetros ajustados."""
