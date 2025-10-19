from typing import Sequence
from point import Point

class Spline:
    def update_spline(self, points: Sequence[Point], f_values: Sequence[float]) -> None:
        raise NotImplementedError

    def get_value(self, p: Point) -> list[float]:
        raise NotImplementedError
