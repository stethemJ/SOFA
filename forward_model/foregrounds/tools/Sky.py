import numpy as np
import numpy.typing as npt
import healpy as hp
from pygdsm import GlobalSkyModel16

from forward_model.foregrounds.tools.config import TE_MAP_PATH, EM_MAP_PATH


class Sky:
    """Per-pixel spectral-index foreground sky (Anstey et al. 2025, arXiv:2607.00299).

    Spectral index β(p) is derived from GSM2016 maps at 230 and 408 MHz.
    Maps at arbitrary frequency are synthesised as T_sync(ν,p) + T_ff(ν,p).
    """

    T_CMB = 2.725   # K
    NU_LO  = 230.0  # MHz — spectral-index lower anchor
    NU_HI  = 408.0  # MHz — spectral-index upper anchor / synchrotron reference

    def __init__(self, nside: int = 64,
                 te_map_path=TE_MAP_PATH, em_map_path=EM_MAP_PATH):
        gsm16 = GlobalSkyModel16(freq_unit="MHz", data_unit="TRJ")
        T_lo = hp.ud_grade(gsm16.generate(self.NU_LO), nside)
        T_hi = hp.ud_grade(gsm16.generate(self.NU_HI), nside)
        self._T_408 = T_hi
        # Per-pixel spectral index. GSM2016 PCA interpolation can produce
        # non-positive values at edge frequencies; fall back to β=2.5 there.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where((T_lo > 0) & (T_hi > 0), T_lo / T_hi, np.nan)
        self._beta = np.where(
            np.isfinite(ratio) & (ratio > 0),
            np.log(ratio) / np.log(self.NU_HI / self.NU_LO),
            2.5,
        )

        self._te = hp.ud_grade(hp.read_map(te_map_path), nside)
        self._em = hp.ud_grade(hp.read_map(em_map_path), nside)

        T_ff_lo = self._free_free(self.NU_LO, self._te, self._em)
        T_fg_lo = np.maximum(T_lo - self.T_CMB, 1e-30)  # guard against near-zero
        self._A_sync = np.clip(1.0 - T_ff_lo / T_fg_lo, 0.0, 1.0)

    def generate(self, frequency: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """T_sync(ν,p) + T_ff(ν,p) HEALPix map(s) [K] at frequency/frequencies (MHz)."""
        freqs = np.atleast_1d(np.asarray(frequency, dtype=float))
        maps = np.stack([self._generate_single(f) for f in freqs])
        return maps[0] if maps.shape[0] == 1 else maps

    def _generate_single(self, nu: float) -> npt.NDArray[np.float64]:
        T_sync = self._A_sync * self._T_408 * (nu / self.NU_HI) ** (-self._beta)
        return T_sync + self._free_free(nu, self._te, self._em)

    @staticmethod
    def _gaunt_factor(nu_mhz: float,
                      te: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Frequency-averaged free-free Gaunt factor (Draine 2011)."""
        nu_ghz = nu_mhz / 1000.0
        inner = 5.960 - (np.sqrt(3) / np.pi) * np.log(nu_ghz * (te / 1e4) ** -1.5)
        return np.log(np.exp(inner) + np.e)

    @staticmethod
    def _free_free(nu_mhz: float,
                   te: npt.NDArray[np.float64],
                   em: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Free-free brightness temperature [K] (Draine 2011 optical-depth formalism)."""
        nu_ghz = nu_mhz / 1000.0
        g_ff = Sky._gaunt_factor(nu_mhz, te)
        tau = 0.05468 * te ** -1.5 * nu_ghz ** -2 * em * g_ff
        return te * (1.0 - np.exp(-tau))
