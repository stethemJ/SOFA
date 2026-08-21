import numpy as np
import numpy.typing as npt
from typing import Sequence
from scipy.constants import c as SPEED_OF_LIGHT

from tools.VshBasis import VshBasis, VshTerm


class EField(VshBasis):
    """An antenna beam pattern (VshBasis) with image-theory ground reflection."""

    def __init__(
        self,
        terms: Sequence[tuple[int, int, str] | VshTerm],
        C: npt.ArrayLike,
        frequency: float,
        height: float,
    ):
        super().__init__(terms, C)
        self.frequency = frequency
        self.height = height

    @classmethod
    def from_basis(cls, basis: VshBasis, frequency: float, height: float) -> "EField":
        """Build an EField from an existing VshBasis's terms/coefficients."""
        return cls(basis.terms, basis.C, frequency, height)

    def apply_image_theory(self, theta: npt.ArrayLike, phi: npt.ArrayLike):
        """Ground-reflected field via image theory: reflect across the xy
        (ground) plane at self.height, at self.frequency. Each term's own
        image sign is derived automatically from its (l, m, mode) via
        VshTerm.image_sign() -- see tools/VshBasis.py and the project plan
        notes for the derivation. Zeroes theta > pi/2 (below ground)."""
        theta = np.asarray(theta, dtype=float)
        phi = np.asarray(phi, dtype=float)
        k = 2 * np.pi * self.frequency / SPEED_OF_LIGHT
        phase = k * self.height * np.cos(theta)
        shape = np.broadcast_shapes(theta.shape, phi.shape)
        e_theta = np.zeros(shape)
        e_phi = np.zeros(shape)
        for c, term in zip(self.C, self.terms):
            array_factor = 2 * np.cos(phase) if term.image_sign() > 0 else 2 * np.sin(phase)
            t_theta, t_phi = term.evaluate(theta, phi)
            e_theta = e_theta + c * array_factor * t_theta
            e_phi = e_phi + c * array_factor * t_phi
        below_ground = theta > np.pi / 2
        e_theta = np.where(below_ground, 0.0, e_theta)
        e_phi = np.where(below_ground, 0.0, e_phi)
        return e_theta, e_phi
