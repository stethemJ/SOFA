import numpy as np
import numpy.typing as npt
from typing import Literal, Sequence
from scipy.constants import c as SPEED_OF_LIGHT

from tools.ShBasis import ShBasis, ShTerm

_POLARIZATION_SIGN = {"vertical": 1.0, "horizontal": -1.0}


class EField(ShBasis):
    """An antenna beam pattern (ShBasis) with image-theory ground reflection."""

    def __init__(
        self,
        terms: Sequence[tuple[int, int] | ShTerm],
        C: npt.ArrayLike,
        frequency: float,
        height: float,
        polarization: Literal["vertical", "horizontal"] = "vertical",
    ):
        super().__init__(terms, C)
        if polarization not in _POLARIZATION_SIGN:
            raise ValueError(f"polarization must be 'vertical' or 'horizontal', got {polarization!r}")
        self.frequency = frequency
        self.height = height
        self.polarization = polarization

    @classmethod
    def from_basis(cls, basis: ShBasis, frequency: float, height: float,
                    polarization: Literal["vertical", "horizontal"] = "vertical") -> "EField":
        """Build an EField from an existing ShBasis's terms/coefficients."""
        return cls(basis.terms, basis.C, frequency, height, polarization)

    def apply_image_theory(self, theta: npt.ArrayLike, phi: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Ground-reflected field via image theory: reflect across the xy
        (ground) plane at self.height, at self.frequency, sign set by
        self.polarization. Zeroes theta > pi/2 (below ground)."""
        theta = np.asarray(theta, dtype=float)
        phi = np.asarray(phi, dtype=float)
        k = 2 * np.pi * self.frequency / SPEED_OF_LIGHT
        phase = k * self.height * np.cos(theta)
        sign = _POLARIZATION_SIGN[self.polarization]
        array_factor = 2 * np.cos(phase) if sign > 0 else 2 * np.sin(phase)
        field = self.evaluate(theta, phi) * array_factor
        return np.where(theta > np.pi / 2, 0.0, field)
