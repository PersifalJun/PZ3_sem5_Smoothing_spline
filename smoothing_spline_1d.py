from typing import Sequence, List
from point import Point
from spline import Spline

class SmoothingSpline1D(Spline):
    def __init__(self, smooth: float) -> None:
        self.smooth = float(smooth)
        self.points: List[Point] = []
        self.alpha: List[float] = []

    def _to_master_ksi(self, seg_num: int, x: float) -> float:
        x0 = self.points[seg_num].x()
        x1 = self.points[seg_num + 1].x()
        return 2.0 * (x - x0) / (x1 - x0) - 1.0

    @staticmethod
    def _basis(number: int, ksi: float) -> float:
        if number == 1: return 0.5 * (1.0 - ksi)
        if number == 2: return 0.5 * (1.0 + ksi)
        raise ValueError("Неверный номер базисной функции")

    @staticmethod
    def _basis_der(number: int) -> float:
        if number == 1: return -0.5
        if number == 2: return  0.5
        raise ValueError("Неверный номер производной базисной функции")

    # weights (список длины len(pts))
    def update_spline(self, pts: Sequence[Point], f_values: Sequence[float], weights: Sequence[float] | None = None) -> None:
        if pts is None or f_values is None or len(pts) != len(f_values) or len(pts) < 2:
            raise ValueError("Некорректные входные данные.")

        self.points = list(pts)
        n_seg = len(self.points) - 1
        self.alpha = [0.0] * (n_seg + 1)

        # веса узлов
        if weights is None:
            w = [1.0] * (n_seg + 1)
        else:
            if len(weights) != n_seg + 1:
                raise ValueError("Длина weights должна совпадать с числом узлов.")
            w = list(weights)

        a = [0.0] * (n_seg + 1)
        b = [0.0] * (n_seg + 1)
        c = [0.0] * (n_seg + 1)

        for i in range(n_seg):

            ksi = self._to_master_ksi(i, self.points[i].x())
            f1 = self._basis(1, ksi); f2 = self._basis(2, ksi)
            wi = w[i]
            b[i]     += (1.0 - self.smooth) * wi * f1 * f1
            b[i + 1] += (1.0 - self.smooth) * wi * f2 * f2
            a[i + 1] += (1.0 - self.smooth) * wi * f1 * f2
            c[i]     += (1.0 - self.smooth) * wi * f2 * f1
            self.alpha[i]     += (1.0 - self.smooth) * wi * f1 * f_values[i]
            self.alpha[i + 1] += (1.0 - self.smooth) * wi * f2 * f_values[i]

            ksi = self._to_master_ksi(i, self.points[i + 1].x())
            f1 = self._basis(1, ksi); f2 = self._basis(2, ksi)
            wi1 = w[i + 1]
            b[i]     += (1.0 - self.smooth) * wi1 * f1 * f1
            b[i + 1] += (1.0 - self.smooth) * wi1 * f2 * f2
            a[i + 1] += (1.0 - self.smooth) * wi1 * f1 * f2
            c[i]     += (1.0 - self.smooth) * wi1 * f2 * f1
            self.alpha[i]     += (1.0 - self.smooth) * wi1 * f1 * f_values[i + 1]
            self.alpha[i + 1] += (1.0 - self.smooth) * wi1 * f2 * f_values[i + 1]


            h = self.points[i + 1].x() - self.points[i].x()
            b[i]     += self.smooth / h
            b[i + 1] += self.smooth / h
            a[i + 1] -= self.smooth / h
            c[i]     -= self.smooth / h

        # прогонка
        for j in range(1, n_seg + 1):
            m = a[j] / b[j - 1]
            b[j]          -= m * c[j - 1]
            self.alpha[j] -= m * self.alpha[j - 1]

        self.alpha[n_seg] /= b[n_seg]
        for j in range(n_seg - 1, -1, -1):
            self.alpha[j] = (self.alpha[j] - self.alpha[j + 1] * c[j]) / b[j]

    def get_value(self, p: Point) -> list[float]:
        eps = 1e-7
        n_seg = len(self.points) - 1
        x = p.x()

        for i in range(n_seg):
            xi = self.points[i].x()
            xj = self.points[i + 1].x()
            if (x > xi and x < xj) or abs(x - xi) < eps or abs(x - xj) < eps:
                h = xj - xi
                ksi = self._to_master_ksi(i, x)
                g  = self.alpha[i] * self._basis(1, ksi) + self.alpha[i + 1] * self._basis(2, ksi)
                gp = (self.alpha[i] * self._basis_der(1) + self.alpha[i + 1] * self._basis_der(2)) * 2.0 / h
                return [g, gp, 0.0]
        raise ValueError("Точка вне диапазона разбиения.")