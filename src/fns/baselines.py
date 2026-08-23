"""
Non-ML reference points.

WHY BASELINES DECIDE WHETHER A PROJECT IS HONEST
------------------------------------------------
US equities drift upward: in this sample 52.4% of next-day returns are positive.
A classifier that has learned nothing and simply always says "up" therefore
scores 52.4% accuracy. Reporting "my model is 52% accurate!" without that number
next to it is the single most common way stock-prediction projects mislead.

Every model in train.py is scored against all three of these.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class AlwaysUpBaseline:
    """Predict 'up' every single day. The bar that actually matters."""

    name = "baseline_always_up"

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        # The "training" is just recording the base rate, which becomes the
        # constant probability we emit -- so AUC is exactly 0.5 by construction.
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.full(len(X), self.p_)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X), dtype=int)


class MovingAverageBaseline:
    """Classic trend following: long when the fast MA is above the slow MA.

    `ma_cross` is already (SMA10 - SMA30) / SMA30, so its SIGN is the crossover
    signal. This is the "simple moving-average baseline" the project brief asks
    for, and it is a fair fight: it uses only price, no news.

    We squash the raw spread through a logistic so the class can also emit a
    probability for AUC. The scale (10) is arbitrary and monotone, so it changes
    the probabilities but never the ranking or the 0/1 predictions.
    """

    name = "baseline_ma_cross"

    def __init__(self, col: str = "ma_cross", scale: float = 10.0):
        self.col, self.scale = col, scale

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        return self          # nothing to learn -- it is a fixed rule

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-self.scale * X[self.col].to_numpy()))
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X[self.col].to_numpy() > 0).astype(int)


class SentimentSignBaseline:
    """Go long when today's average news tone is positive.

    This is the naive reading of the project premise -- 'good news, buy'. Keeping
    it as an explicit baseline lets the report state precisely how much the
    trained model adds over simply trusting FinBERT's sign.
    """

    name = "baseline_sentiment_sign"

    def __init__(self, col: str = "sent_mean", scale: float = 2.0):
        self.col, self.scale = col, scale

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-self.scale * X[self.col].to_numpy()))
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X[self.col].to_numpy() > 0).astype(int)


ALL_BASELINES = (AlwaysUpBaseline, MovingAverageBaseline, SentimentSignBaseline)
