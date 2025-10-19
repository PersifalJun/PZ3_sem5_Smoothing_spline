from __future__ import annotations
import time
import statistics as stats
import numpy as np

from lib_spline_utils import build_normal_sample, cubic_interp_lib, smoothing_lib

from point import Point
from cubic_interpolation_spline_1d import CubicInterpolationSpline1D
from smoothing_spline_1d import SmoothingSpline1D

P_LIST = [0.0, 0.4, 0.8, 0.99]

def build_points_list(x: np.ndarray):
    return [Point(float(xi), 0.0, 0.0) for xi in x]

def timeit(fn, repeat=3):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
    return stats.mean(times), stats.stdev(times) if repeat > 1 else 0.0

def bench_case(N: int, mean: float, sigma: float):
    x, y = build_normal_sample(N, mean, sigma, seed=242025)
    pts = build_points_list(x)
    fvals = list(map(float, y))
    w_all = [1.0] * N

    def run_interp():
        spl = CubicInterpolationSpline1D()
        spl.update_spline(pts, fvals)
        _ = [spl.get_value(p)[0] for p in pts]

    def run_smooth():
        for p in P_LIST:
            sm = SmoothingSpline1D(p)
            sm.update_spline(pts, fvals, weights=w_all)
            _ = [sm.get_value(pnt)[0] for pnt in pts]

    t_y_interp, s_y_interp = timeit(run_interp)
    t_y_smooth, s_y_smooth = timeit(run_smooth)

    def run_lib_interp():
        spl = cubic_interp_lib(x, y)
        _ = spl(x)

    def run_lib_smooth():
        sm = smoothing_lib(x, y, np.ones_like(y), P_LIST)
        _ = [sm[p](x) for p in P_LIST]

    t_l_interp, s_l_interp = timeit(run_lib_interp)
    t_l_smooth, s_l_smooth = timeit(run_lib_smooth)

    print(f"N={N}:")
    print(f"  Интерполяция без готовых решений : {t_y_interp:.2f} ± {s_y_interp:.2f} ms")
    print(f"  Сглаживание без готовых решений : {t_y_smooth:.2f} ± {s_y_smooth:.2f} ms")
    print(f"  SciPy интерполяция : {t_l_interp:.2f} ± {s_l_interp:.2f} ms")
    print(f"  SciPy сглаживание : {t_l_smooth:.2f} ± {s_l_smooth:.2f} ms")
    print()

def main():
    for N in (10, 100, 1087):
        bench_case(N, mean=1.08, sigma=4.96)

if __name__ == "__main__":
    main()
