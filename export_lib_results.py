from __future__ import annotations
import csv
import numpy as np
from lib_spline_utils import build_normal_sample, cubic_interp_lib, smoothing_lib

N, MEAN, SIGMA = 1087, 1.08, 4.96
P_LIST = [0.0, 0.4, 0.8, 0.99]

def save_csv(filename: str, x: np.ndarray, y: np.ndarray, w: np.ndarray,
             y_interp: np.ndarray, smooth_dict: dict[float, np.ndarray]) -> None:
    header = ["index", "value_f", "weight_w", "g_interp_lib"] + [f"g_p_{p:.2f}_lib" for p in P_LIST]
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f, delimiter=';')
        wr.writerow(header)
        for i in range(len(x)):
            row = [i + 1, float(y[i]), float(w[i]), float(y_interp[i])]
            for p in P_LIST:
                row.append(float(smooth_dict[p][i]))
            wr.writerow(row)
    print(f"Saved: {filename}")

def main():
    # данные
    x, y = build_normal_sample(N, MEAN, SIGMA, seed=242025)

    w1 = np.ones_like(y)
    interp = cubic_interp_lib(x, y)
    y_interp = interp(x)

    smooth = smoothing_lib(x, y, w1, P_LIST)
    smooth_vals = {p: smooth[p](x) for p in P_LIST}
    save_csv("spline_lib_w1.csv", x, y, w1, y_interp, smooth_vals)


    w2 = np.ones_like(y)              # <-- ЭТОГО НЕ ХВАТАЛО
    for k in [100,200,300,400,500,600,700,800,900,1000]:
        if 1 <= k <= len(y):
            w2[k-1] = 0.5

    smooth2 = smoothing_lib(x, y, w2, P_LIST)
    smooth_vals2 = {p: smooth2[p](x) for p in P_LIST}
    save_csv("spline_lib_w_half.csv", x, y, w2, y_interp, smooth_vals2)

if __name__ == "__main__":
    main()
