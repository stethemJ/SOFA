import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from math import factorial
from typing import Sequence
from scipy.special import lpmv


def _validate_lm(l: int, m: int) -> None:
    if l < 0:
        raise ValueError(f"l must be >= 0, got {l}")
    if abs(m) > l:
        raise ValueError(f"require |m| <= l, got l={l}, m={m}")


def real_sph_harm(l: int, m: int, theta: npt.ArrayLike, phi: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Real orthonormal spherical harmonic Y_l^m(theta, phi).

    theta: polar/colatitude angle in [0, pi]. phi: azimuthal angle in [0, 2*pi).
    Broadcasts elementwise; pass np.meshgrid(..., indexing="ij") outputs for a grid.
    """
    _validate_lm(l, m)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    am = abs(m)
    N = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - am) / factorial(l + am))
    P = lpmv(am, l, np.cos(theta)) * (-1) ** am
    if m == 0:
        return N * P * np.ones_like(phi)
    if m > 0:
        return np.sqrt(2) * N * P * np.cos(m * phi)
    return np.sqrt(2) * N * P * np.sin(am * phi)


@dataclass(frozen=True)
class ShTerm:
    l: int
    m: int

    def __post_init__(self) -> None:
        _validate_lm(self.l, self.m)

    def evaluate(self, theta: npt.ArrayLike, phi: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return real_sph_harm(self.l, self.m, theta, phi)


class ShBasis:
    """A linear combination of real Ylm basis terms with coefficient vector C."""

    def __init__(self, terms: Sequence[tuple[int, int] | ShTerm], C: npt.ArrayLike):
        self.terms: list[ShTerm] = [t if isinstance(t, ShTerm) else ShTerm(*t) for t in terms]
        self.C = np.atleast_1d(np.asarray(C, dtype=float))
        if len(self.terms) != len(self.C):
            raise ValueError(
                f"terms and C length mismatch: {len(self.terms)} terms vs {len(self.C)} coefficients"
            )

    def evaluate(self, theta: npt.ArrayLike, phi: npt.ArrayLike) -> npt.NDArray[np.float64]:
        theta = np.asarray(theta, dtype=float)
        phi = np.asarray(phi, dtype=float)
        result = np.zeros(np.broadcast_shapes(theta.shape, phi.shape))
        for c, term in zip(self.C, self.terms):
            result = result + c * term.evaluate(theta, phi)
        return result

    def add_term(self, l: int, m: int, c: float) -> "ShBasis":
        return ShBasis(self.terms + [ShTerm(l, m)], np.append(self.C, c))

    def __add__(self, other: "ShBasis") -> "ShBasis":
        return ShBasis(self.terms + other.terms, np.concatenate([self.C, other.C]))

    def __len__(self) -> int:
        return len(self.terms)

    def __repr__(self) -> str:
        parts = ", ".join(f"{c:g}*Y_{t.l}^{t.m}" for c, t in zip(self.C, self.terms))
        return f"ShBasis({parts})"
