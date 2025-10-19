from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    x_: float = 0.0
    y_: float = 0.0
    z_: float = 0.0

    def x(self) -> float: return self.x_

    def y(self) -> float: return self.y_

    def z(self) -> float: return self.z_
