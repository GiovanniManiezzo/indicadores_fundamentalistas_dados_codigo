"""LSTM: Rede neural LSTM parcimonioso (experimental).

Arquitetura ultra-parcimonial obrigatória (N=64 → risco de overfitting):
  - 1 camada LSTM × 8 unidades
  - lookback = 4 passos
  - Regularização L2 em kernel e recurrent weights
  - Sem exógenas (modelo AR puro)

Requer tensorflow >= 2.0. Levanta ImportError se indisponível.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from .base_model import BaseModel

_TF_AVAILABLE = False
try:
    import tensorflow as _tf  # noqa: F401
    _TF_AVAILABLE = True
except (ImportError, OSError):
    pass


class LSTMWrapper(BaseModel):
    """LSTM AR puro (sem exógenas). Experimental."""

    def __init__(
        self,
        lookback: int = 4,
        units: int = 8,
        epochs: int = 100,
        l2: float = 0.01,
        mode: str = "raw",
    ) -> None:
        if not _TF_AVAILABLE:
            raise ImportError(
                "tensorflow não instalado — LSTMWrapper indisponível"
            )
        self.lookback = lookback
        self.units = units
        self.epochs = epochs
        self.l2 = l2
        self.mode = mode
        self._model = None
        self._scaler_mean: float = 0.0
        self._scaler_std: float = 1.0
        self._last_window: np.ndarray | None = None  # (lookback, 1) normalizado
        self._y_original: pd.Series | None = None

    def _transform(self, y: pd.Series) -> pd.Series:
        mode = self.mode
        if mode == "raw":
            return y
        if mode == "log":
            if (y <= 0).any():
                raise ValueError(f"log requer valores positivos; mín={y.min():.4g}")
            return np.log(y)
        if mode == "diff":
            return y.diff().dropna()
        if mode == "log+diff":
            if (y <= 0).any():
                raise ValueError(f"log+diff requer valores positivos; mín={y.min():.4g}")
            return np.log(y).diff().dropna()
        raise ValueError(f"Modo desconhecido: {mode!r}")

    def fit(self, y: pd.Series, u: pd.DataFrame | None = None) -> "LSTMWrapper":
        import tensorflow as tf
        from tensorflow.keras import regularizers  # type: ignore[import]

        self._y_original = y.copy()
        y_t = self._transform(y)
        yv = y_t.values.astype(float)

        # Normalização z-score
        self._scaler_mean = float(np.mean(yv))
        self._scaler_std  = float(np.std(yv)) or 1.0
        yv_n = (yv - self._scaler_mean) / self._scaler_std

        # Janelas deslizantes
        X_rows, Y_vals = [], []
        for t in range(self.lookback, len(yv_n)):
            X_rows.append(yv_n[t - self.lookback:t])
            Y_vals.append(yv_n[t])

        if len(X_rows) < 3:
            raise RuntimeError(
                f"LSTM: amostras insuficientes (n={len(X_rows)}) para lookback={self.lookback}"
            )

        X = np.array(X_rows, dtype=float).reshape(-1, self.lookback, 1)
        Y = np.array(Y_vals, dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tf.keras.backend.clear_session()
            tf.random.set_seed(42)
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.lookback, 1)),
                tf.keras.layers.LSTM(
                    self.units,
                    kernel_regularizer=regularizers.l2(self.l2),
                    recurrent_regularizer=regularizers.l2(self.l2),
                ),
                tf.keras.layers.Dense(1),
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(
                X, Y,
                epochs=self.epochs,
                batch_size=max(4, min(16, len(X))),
                verbose=0,
            )

        self._model = model
        self._last_window = yv_n[-self.lookback:].reshape(1, self.lookback, 1)
        return self

    def predict(self, steps: int = 1, u_future: pd.DataFrame | None = None) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred_n = float(self._model.predict(self._last_window, verbose=0).flatten()[0])

        # Desnormalizar
        raw = np.array([pred_n * self._scaler_std + self._scaler_mean], dtype=float)

        mode = self.mode
        if mode == "raw":
            return raw
        if mode == "log":
            return np.exp(raw)
        if mode == "diff":
            last = float(self._y_original.iloc[-1])
            return np.cumsum(raw) + last
        if mode == "log+diff":
            last = float(self._y_original.iloc[-1])
            log_last = np.log(last) if last > 0 else 0.0
            return np.exp(np.cumsum(raw) + log_last)
        raise ValueError(f"Modo desconhecido: {mode!r}")

    def get_params_str(self) -> str:
        return f"LSTM(lb={self.lookback},units={self.units})"
