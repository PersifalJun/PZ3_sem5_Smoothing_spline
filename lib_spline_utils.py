from __future__ import annotations
from typing import List, Sequence, Dict
import numpy as np

try:
    from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
except Exception as e:
    raise RuntimeError(
        "Ошибка использования библиотеки scipy!\n"
        f"Подробности: {e}"
    )


def build_normal_sample(N: int, mean: float, sigma: float, seed: int = 242025):
    rnd = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, N, dtype=np.float64)  # узлы [0,1]
    y = rnd.normal(loc=mean, scale=sigma, size=N).astype(np.float64)
    return x, y


def cubic_interp_lib(x: np.ndarray, y: np.ndarray):
    return InterpolatedUnivariateSpline(x, y, k=3)


def smoothing_lib(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None,
    p_list: Sequence[float],
) -> Dict[float, UnivariateSpline]:

    if w is None:
        w = np.ones_like(y)
    w = np.asarray(w, dtype=np.float64)


    y_mean = np.average(y, weights=w)
    Smax = float(np.sum(w * (y - y_mean) ** 2))

    result: Dict[float, UnivariateSpline] = {}
    for p in p_list:
        p = float(p)
        s = max(0.0, min(1.0, p)) * Smax
        spl = UnivariateSpline(x, y, w=w, s=s, k=3)
        result[p] = spl
    return result
