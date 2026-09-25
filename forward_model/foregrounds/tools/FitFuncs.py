import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares


class FitFuncs:
    """Two foreground spectral models fit to sky temperature vs. frequency (MHz).

    log_poly   — phenomenological log-space polynomial (linear lstsq).
    sync_ff    — physical synchrotron + free-free model (Draine 2011, nonlinear).
    """

    def __init__(self, nu_ref: float, T_ref: float):
        self.nu_ref = nu_ref  # reference frequency [MHz]
        self.T_ref = T_ref    # reference temperature [K], used only for log_poly normalisation

    # ------------------------------------------------------------------
    # Log-space polynomial  T = T_ref * sum_{n=0}^{N} a_n * ln(nu/nu_ref)^n
    # ------------------------------------------------------------------

    def log_poly(self, nu: npt.ArrayLike, coeffs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        nu = np.asarray(nu, dtype=float)
        x = np.log(nu / self.nu_ref)
        return self.T_ref * sum(a * x**n for n, a in enumerate(np.asarray(coeffs, dtype=float)))

    def fit_log_poly(self, nu: npt.ArrayLike, T: npt.ArrayLike, N: int) -> npt.NDArray[np.float64]:
        """Linear least squares — returns (N+1,) coefficient array."""
        nu = np.asarray(nu, dtype=float)
        x = np.log(nu / self.nu_ref)
        design = np.vstack([self.T_ref * x**n for n in range(N + 1)]).T
        coeffs, *_ = np.linalg.lstsq(design, np.asarray(T, dtype=float), rcond=None)
        return coeffs

    # ------------------------------------------------------------------
    # Synchrotron + free-free   (Draine 2011 optical-depth formalism)
    # params = [A_s, beta, T_e, EM]
    #   A_s  — synchrotron amplitude at nu_ref [K]
    #   beta — synchrotron spectral index (> 0; typically 2.5–3.0)
    #   T_e  — electron temperature [K]
    #   EM   — emission measure [pc cm^-6]
    # ------------------------------------------------------------------

    @staticmethod
    def _gaunt_factor(nu_mhz: float, te: float) -> float:
        nu_ghz = nu_mhz / 1000.0
        inner = 5.960 - (np.sqrt(3) / np.pi) * np.log(nu_ghz * (te / 1e4) ** -1.5)
        return np.log(np.exp(inner) + np.e)

    def _T_ff(self, nu: npt.NDArray[np.float64], te: float, em: float) -> npt.NDArray[np.float64]:
        nu_ghz = nu / 1000.0
        g_ff = self._gaunt_factor(nu, te)
        tau = 0.05468 * te**-1.5 * nu_ghz**-2 * em * g_ff
        return te * (1.0 - np.exp(-tau))

    def sync_ff(self, nu: npt.ArrayLike, params: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """T_sync(nu) + T_ff(nu) given params = [A_s, beta, T_e, EM]."""
        nu = np.asarray(nu, dtype=float)
        A_s, beta, te, em = params
        T_sync = A_s * (nu / self.nu_ref) ** (-beta)
        return T_sync + self._T_ff(nu, te, em)

    def fit_sync_ff(self, nu: npt.ArrayLike, T: npt.ArrayLike,
                    x0: npt.ArrayLike = None) -> npt.NDArray[np.float64]:
        """Nonlinear least squares — returns params = [A_s, beta, T_e, EM].

        Default initial guess: A_s=T_ref, beta=2.5, T_e=8000 K, EM=1 pc/cm^6.
        Bounds enforce positivity and physically reasonable ranges.
        """
        nu = np.asarray(nu, dtype=float)
        T = np.asarray(T, dtype=float)
        if x0 is None:
            x0 = [self.T_ref, 2.5, 8000.0, 1.0]
        bounds = (
            [0.0,  1.0,  100.0,  0.0],   # lower
            [np.inf, 5.0, 3e4,  1e4],    # upper
        )
        res = least_squares(
            lambda p: T - self.sync_ff(nu, p),
            x0,
            bounds=bounds,
            method="trf",
        )
        return res.x
