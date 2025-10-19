import csv
import random
from typing import List, Sequence

from point import Point
from cubic_interpolation_spline_1d import CubicInterpolationSpline1D
from smoothing_spline_1d import SmoothingSpline1D


N, MEAN, SIGMA = 1087, 1.08, 4.96
P_LIST = [0.0, 0.4, 0.8, 0.99]

#генерация данных
def build_normal_sample(N: int, mean: float, sigma: float):
    rnd = random.Random(24_2025)  # фиксированный seed
    pts: List[Point] = []
    fvals: List[float] = []
    for i in range(N):
        x = i / float(N - 1)
        pts.append(Point(x, 0.0, 0.0))
        fvals.append(rnd.gauss(mean, sigma))
    return pts, fvals

#печать
def sci(x: float) -> str:
    return f"{x:.6E}"

def simple_box_header(N: int, M: float, S: float) -> None:
    print("| {:^18} | {:^18} | {:^18} |".format("Число наблюдений N", "Мат. ожидание M", "Отклонение σ"))
    print("| {:>18} | {:>18} | {:>18} |".format(N, f"{M:.2f}", f"{S:.2f}"))
    print("+--------------------+--------------------+--------------------+\n")

def print_interp_table(fvals: Sequence[float], weights: Sequence[float], interp_vals: Sequence[float]) -> None:
    w_idx, w_val, w_w, w_g = 5, 16, 5, 16
    print("Интерполяционный сплайн")
    head = f"{'№':>{w_idx}} | {'Случайная величина':^{w_val}} | {'Вес w':^{w_w}} | {'g_interp (значение интерполяционного сплайна в узле)':^{w_g}}"
    print(head)
    print("-" * len(head))
    for i, (y, w, g) in enumerate(zip(fvals, weights, interp_vals), start=1):
        print(f"{i:>{w_idx}} | {sci(y):>{w_val}} | {w:^{w_w}.3g} | {sci(g):>{w_g}}")
    print()

def print_smoothing_table(fvals: Sequence[float], weights: Sequence[float],
                          p_list: Sequence[float], smooth_vals_by_p: Sequence[Sequence[float]]) -> None:
    w_idx, w_val, w_w, w_p = 5, 16, 5, 14
    print("Сглаживающий сплайн")
    head = f"{'№':>{w_idx}} | {'Случайная величина':^{w_val}} | {'Вес w':^{w_w}}"
    for p in p_list:
        head += f" | {'p = ' + f'{p:.2f}':^{w_p}}"
    print(head)
    print("-" * len(head))
    for i, (y, w) in enumerate(zip(fvals, weights), start=1):
        row = f"{i:>{w_idx}} | {sci(y):>{w_val}} | {w:^{w_w}.3g}"
        for col in smooth_vals_by_p:
            row += f" | {sci(col[i-1]):>{w_p}}"
        print(row)
    print()

def main() -> None:
    pts, fvals = build_normal_sample(N, MEAN, SIGMA)
    weights = [1.0] * len(pts)

    #интерполяционный сплайн
    cubic = CubicInterpolationSpline1D()
    cubic.update_spline(pts, fvals)
    g_interp = [cubic.get_value(p)[0] for p in pts]

    #сглаживающий сплайн
    smooth_cols: List[List[float]] = []
    for p in P_LIST:
        sm = SmoothingSpline1D(p)
        sm.update_spline(pts, fvals, weights=weights)
        smooth_cols.append([sm.get_value(pnt)[0] for pnt in pts])

    simple_box_header(N, MEAN, SIGMA)
    print_interp_table(fvals, weights, g_interp)
    print_smoothing_table(fvals, weights, P_LIST, smooth_cols)

    #сохранение в csv
    out_name = "spline_output.csv"
    with open(out_name, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=';')  # ; удобнее для Excel (RU)
        header = ["index", "value_f", "weight_w", "g_interp"] + [f"g_p_{p:.2f}" for p in P_LIST]
        writer.writerow(header)
        for i in range(len(pts)):
            row = [i + 1, fvals[i], weights[i], g_interp[i]] + [smooth_cols[j][i] for j in range(len(P_LIST))]
            writer.writerow(row)

    print(f"Saved: {out_name}")


    weights_half = [1.0] * len(pts)
    weak_idx = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]  # 1-based
    for k in weak_idx:
        if 1 <= k <= len(weights_half):
            weights_half[k - 1] = 0.5


    smooth_cols_half: List[List[float]] = []
    for p in P_LIST:
        sm = SmoothingSpline1D(p)
        sm.update_spline(pts, fvals, weights=weights_half)
        smooth_cols_half.append([sm.get_value(pnt)[0] for pnt in pts])

    # сохранение второго CSV
    out_name2 = "spline_output_w_half.csv"
    with open(out_name2, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=';')
        header = ["index", "value_f", "weight_w", "g_interp"] + [f"g_p_{p:.2f}" for p in P_LIST]
        writer.writerow(header)
        for i in range(len(pts)):
            row = [i + 1, fvals[i], weights_half[i], g_interp[i]] \
                  + [smooth_cols_half[j][i] for j in range(len(P_LIST))]
            writer.writerow(row)
    print(f"Saved: {out_name2}")

if __name__ == "__main__":
    main()
