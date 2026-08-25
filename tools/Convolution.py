import numpy as np
import numpy.typing as npt
import healpy as hp

from tools.ShBasis import real_sph_harm
from tools.EField import EField

DEFAULT_L_MAX = 20
DEFAULT_BEAM_NSIDE = 64

_design_matrix_cache = {}  # (l_max, nside) -> (matrix, theta, phi, pixel_area)


def _get_design_matrix(l_max: int, nside: int):
    key = (l_max, nside)
    if key not in _design_matrix_cache:
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        pixel_area = 4 * np.pi / npix
        n_harmonics = (l_max + 1) ** 2
        matrix = np.empty((n_harmonics, npix))
        idx = 0
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                matrix[idx] = real_sph_harm(l, m, theta, phi)
                idx += 1
        _design_matrix_cache[key] = (matrix, theta, phi, pixel_area)
    return _design_matrix_cache[key]


def get_grid(l_max: int = DEFAULT_L_MAX, nside: int = DEFAULT_BEAM_NSIDE):
    """(theta, phi) grid used for projection at this (l_max, nside)."""
    _, theta, phi, _ = _get_design_matrix(l_max, nside)
    return theta, phi


def project_to_ylm(values: npt.ArrayLike, l_max: int = DEFAULT_L_MAX,
                    nside: int = DEFAULT_BEAM_NSIDE) -> npt.NDArray[np.float64]:
    """Project a scalar function sampled on this (l_max, nside)'s fixed
    grid onto real orthonormal Y_lm (matrix-multiply against a cached
    design matrix -- fast for repeated calls at the same grid). Returns
    a flat vector of length (l_max+1)^2, canonically ordered l=0..l_max,
    m=-l..l within each l."""
    matrix, _, _, pixel_area = _get_design_matrix(l_max, nside)
    values = np.asarray(values, dtype=float)
    return matrix @ values * pixel_area


def beam_power_coefficients(efield: EField, l_max: int = DEFAULT_L_MAX,
                             nside: int = DEFAULT_BEAM_NSIDE) -> npt.NDArray[np.float64]:
    """Scalar Y_lm coefficients of the beam's |E|^2 power pattern (image
    theory applied), evaluated on a fixed healpix grid."""
    theta, phi = get_grid(l_max, nside)
    e_theta, e_phi = efield.apply_image_theory(theta, phi)
    beam_power = e_theta**2 + e_phi**2
    return project_to_ylm(beam_power, l_max, nside)


def sky_coefficients(sky_map: npt.ArrayLike, l_max: int = DEFAULT_L_MAX,
                      nside: int = DEFAULT_BEAM_NSIDE) -> npt.NDArray[np.float64]:
    """Scalar Y_lm coefficients of a healpix sky map, downgraded to
    `nside` first for speed (safe since l_max << nside)."""
    sky_map = hp.ud_grade(np.asarray(sky_map, dtype=float), nside)
    return project_to_ylm(sky_map, l_max, nside)


def antenna_temperature(beam_coeffs: npt.NDArray[np.float64],
                         sky_coeffs: npt.NDArray[np.float64]) -> float:
    """Beam-weighted sky integral (Parseval dot product), normalized by
    the beam's own solid-angle integral."""
    beam_00 = beam_coeffs[0]
    return float(np.dot(beam_coeffs, sky_coeffs) / (beam_00 * np.sqrt(4 * np.pi)))


def convolve(efield: EField, sky_map: npt.ArrayLike, l_max: int = DEFAULT_L_MAX,
             nside: int = DEFAULT_BEAM_NSIDE) -> float:
    """Convenience: beam-weighted antenna temperature for one EField beam
    and one sky map, in a single call."""
    beam_coeffs = beam_power_coefficients(efield, l_max, nside)
    sky_coeffs_ = sky_coefficients(sky_map, l_max, nside)
    return antenna_temperature(beam_coeffs, sky_coeffs_)
