from typing import Sequence, List
from point import Point
from spline import Spline

class CubicInterpolationSpline1D(Spline):
    def __init__(self) -> None:
        self.points: List[Point] = []
        self.a: List[float] = []
        self.b: List[float] = []
        self.c: List[float] = []
        self.d: List[float] = []

    def update_spline(self, pts: Sequence[Point], f_values: Sequence[float]) -> None:
        if pts is None or f_values is None or len(pts) != len(f_values) or len(pts) < 2:
            raise ValueError("Некорректные входные данные.")

        self.points = list(pts)
        num_seg = len(self.points) - 1
        self.a = [0.0] * num_seg
        self.b = [0.0] * num_seg
        self.c = [0.0] * num_seg
        self.d = [0.0] * num_seg

        if num_seg - 1 <= 0:
            h = self.points[1].x() - self.points[0].x()
            self.a[0] = f_values[0]
            self.b[0] = (f_values[1] - f_values[0]) / h
            self.c[0] = 0.0
            self.d[0] = 0.0
            return

        rhs = [0.0] * (num_seg - 1)

        for i in range(num_seg - 1):
            h_cur = self.points[i + 1].x() - self.points[i].x()
            h_next = self.points[i + 2].x() - self.points[i + 1].x()
            self.b[i] = 2.0 * (h_cur + h_next)
            if i + 1 < num_seg:
                self.a[i + 1] = h_cur
            self.d[i] = h_next

            df_next = (f_values[i + 2] - f_values[i + 1]) / h_next
            df_cur  = (f_values[i + 1] - f_values[i]) / h_cur
            rhs[i] = 3.0 * (df_next - df_cur)

        for j in range(1, num_seg - 1):
            m = self.a[j] / self.b[j - 1]
            self.b[j] -= m * self.d[j - 1]
            rhs[j]    -= m * rhs[j - 1]

        self.c[num_seg - 1] = rhs[num_seg - 2] / self.b[num_seg - 2]
        for j in range(num_seg - 2, 0, -1):
            self.c[j] = (rhs[j - 1] - self.c[j + 1] * self.d[j - 1]) / self.b[j - 1]
        self.c[0] = 0.0

        for i in range(num_seg - 1):
            h = self.points[i + 1].x() - self.points[i].x()
            self.a[i] = f_values[i]
            self.b[i] = (f_values[i + 1] - f_values[i]) / h - (self.c[i + 1] + 2.0 * self.c[i]) * h / 3.0
            self.d[i] = (self.c[i + 1] - self.c[i]) / (3.0 * h)

        h_last = self.points[num_seg].x() - self.points[num_seg - 1].x()
        self.a[num_seg - 1] = f_values[num_seg - 1]
        self.b[num_seg - 1] = (f_values[num_seg] - f_values[num_seg - 1]) / h_last - 2.0 * self.c[num_seg - 1] * h_last / 3.0
        self.d[num_seg - 1] = -self.c[num_seg - 1] / (3.0 * h_last)

    def get_value(self, p: Point) -> list[float]:
        eps = 1e-7
        num_seg = len(self.points) - 1
        x = p.x()

        for i in range(num_seg):
            xi = self.points[i].x()
            xj = self.points[i + 1].x()
            if (x > xi and x < xj) or abs(x - xi) < eps or abs(x - xj) < eps:
                diff = x - xi
                g   = self.a[i] + self.b[i] * diff + self.c[i] * diff**2 + self.d[i] * diff**3
                gp  = self.b[i] + 2.0 * self.c[i] * diff + 3.0 * self.d[i] * diff**2
                gpp = 2.0 * self.c[i] + 6.0 * self.d[i] * diff
                return [g, gp, gpp]
        raise ValueError("Точка вне диапазона разбиения.")
