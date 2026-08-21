import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Sequence

from tools.ShBasis import real_sph_harm, _validate_lm

_DTHETA = 1e-5


def _dY_dtheta(l: int, m: int, theta: npt.ArrayLike, phi: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Central finite difference in theta, with an adaptive step near the
    poles (theta=0 or pi). A fixed step would let theta +/- _DTHETA cross
    past 0 or pi; since real_sph_harm depends on theta only through
    cos(theta) (even), a stencil point that crosses a pole silently
    aliases to its mirror image instead of erroring, corrupting the
    derivative for theta closer to a pole than _DTHETA. Shrinking the step
    to stay strictly inside (0, pi) avoids that."""
    theta = np.asarray(theta, dtype=float)
    step = np.minimum(_DTHETA, np.minimum(theta, np.pi - theta) / 2)
    return (real_sph_harm(l, m, theta + step, phi)
            - real_sph_harm(l, m, theta - step, phi)) / (2 * step)


def _dY_dphi(l: int, m: int, theta: npt.ArrayLike, phi: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """dY_lm/dphi = -m * Y_{l,-m}, exact for all m (including m=0, trivially 0)."""
    if m == 0:
        theta = np.asarray(theta, dtype=float)
        phi = np.asarray(phi, dtype=float)
        return np.zeros(np.broadcast_shapes(theta.shape, phi.shape))
    return -m * real_sph_harm(l, -m, theta, phi)


def vsh_gradient(l: int, m: int, theta: npt.ArrayLike, phi: npt.ArrayLike):
    """Gradient-type ("TM") real VSH B_lm: (E_theta, E_phi)."""
    if l < 1:
        raise ValueError(f"VSH requires l >= 1 (no vector monopole), got l={l}")
    _validate_lm(l, m)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    norm = 1 / np.sqrt(l * (l + 1))
    e_theta = norm * _dY_dtheta(l, m, theta, phi)
    e_phi = norm * _dY_dphi(l, m, theta, phi) / np.sin(theta)
    return e_theta, e_phi


def vsh_curl(l: int, m: int, theta: npt.ArrayLike, phi: npt.ArrayLike):
    """Curl-type ("TE") real VSH C_lm: (E_theta, E_phi) = (-B_phi, B_theta)."""
    b_theta, b_phi = vsh_gradient(l, m, theta, phi)
    return -b_phi, b_theta


@dataclass(frozen=True)
class VshTerm:
    l: int
    m: int
    mode: str  # "TM" (gradient) or "TE" (curl)

    def __post_init__(self) -> None:
        if self.l < 1:
            raise ValueError(f"VSH requires l >= 1 (no vector monopole), got l={self.l}")
        _validate_lm(self.l, self.m)
        if self.mode not in ("TM", "TE"):
            raise ValueError(f"mode must be 'TM' or 'TE', got {self.mode!r}")

    def evaluate(self, theta: npt.ArrayLike, phi: npt.ArrayLike):
        fn = vsh_gradient if self.mode == "TM" else vsh_curl
        return fn(self.l, self.m, theta, phi)

    def image_sign(self) -> float:
        """Ground-plane (PEC) image-theory sign for this (l,m,mode) term's
        far-field pattern: Gamma_TM = (-1)^(l+m+1), Gamma_TE = (-1)^(l+m).
        Derived from multipole parity under spatial inversion combined with
        the (-1)^m rotation-by-pi factor and the PEC image current's extra
        sign flip (electric/magnetic duality: TE and TM always have opposite
        sign at the same (l,m))."""
        parity = (self.l + self.m) % 2
        if self.mode == "TM":
            return -1.0 if parity == 0 else 1.0
        return 1.0 if parity == 0 else -1.0


class VshBasis:
    """A linear combination of real vector spherical harmonic (VSH) terms
    with coefficient vector C. evaluate() returns (E_theta, E_phi) -- a
    genuine vector field, not a scalar."""

    def __init__(self, terms: Sequence[tuple[int, int, str] | VshTerm], C: npt.ArrayLike):
        self.terms: list[VshTerm] = [t if isinstance(t, VshTerm) else VshTerm(*t) for t in terms]
        self.C = np.atleast_1d(np.asarray(C, dtype=float))
        if len(self.terms) != len(self.C):
            raise ValueError(
                f"terms and C length mismatch: {len(self.terms)} terms vs {len(self.C)} coefficients"
            )

    def evaluate(self, theta: npt.ArrayLike, phi: npt.ArrayLike):
        theta = np.asarray(theta, dtype=float)
        phi = np.asarray(phi, dtype=float)
        shape = np.broadcast_shapes(theta.shape, phi.shape)
        e_theta = np.zeros(shape)
        e_phi = np.zeros(shape)
        for c, term in zip(self.C, self.terms):
            t_theta, t_phi = term.evaluate(theta, phi)
            e_theta = e_theta + c * t_theta
            e_phi = e_phi + c * t_phi
        return e_theta, e_phi

    def add_term(self, l: int, m: int, mode: str, c: float) -> "VshBasis":
        return VshBasis(self.terms + [VshTerm(l, m, mode)], np.append(self.C, c))

    def __add__(self, other: "VshBasis") -> "VshBasis":
        return VshBasis(self.terms + other.terms, np.concatenate([self.C, other.C]))

    def __len__(self) -> int:
        return len(self.terms)

    def __repr__(self) -> str:
        parts = ", ".join(f"{c:g}*{t.mode}_{t.l}^{t.m}" for c, t in zip(self.C, self.terms))
        return f"VshBasis({parts})"
