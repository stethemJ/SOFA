import numpy as np
import numpy.typing as npt


class FitFuncs:
    """Foreground spectral models fit to sky temperature vs. frequency
    (MHz), plus linear-least-squares fitting for each (all three forms
    are linear in their coefficients)."""

    def __init__(self, nu_ref: float, T_ref: float):
        self.nu_ref = nu_ref
        self.T_ref = T_ref

    # --- log_poly: T = T_ref * sum_{n=0}^{N} a_n * ln(nu/nu_ref)^n ---

    def log_poly(self, nu: npt.ArrayLike, coeffs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        nu = np.asarray(nu, dtype=float)
        x = np.log(nu / self.nu_ref)
        coeffs = np.asarray(coeffs, dtype=float)
        return self.T_ref * sum(a * x**n for n, a in enumerate(coeffs))

    def fit_log_poly(self, nu: npt.ArrayLike, T: npt.ArrayLike, N: int) -> npt.NDArray[np.float64]:
        nu = np.asarray(nu, dtype=float)
        x = np.log(nu / self.nu_ref)
        design = np.vstack([self.T_ref * x**n for n in range(N + 1)]).T
        coeffs, *_ = np.linalg.lstsq(design, np.asarray(T, dtype=float), rcond=None)
        return coeffs

    # --- power_poly: T = sum_{n=0}^{N-1} a_n * nu^(n-2.5) ---

    def power_poly(self, nu: npt.ArrayLike, coeffs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        nu = np.asarray(nu, dtype=float)
        coeffs = np.asarray(coeffs, dtype=float)
        return sum(a * nu ** (n - 2.5) for n, a in enumerate(coeffs))

    def fit_power_poly(self, nu: npt.ArrayLike, T: npt.ArrayLike, N: int) -> npt.NDArray[np.float64]:
        nu = np.asarray(nu, dtype=float)
        design = np.vstack([nu ** (n - 2.5) for n in range(N)]).T
        coeffs, *_ = np.linalg.lstsq(design, np.asarray(T, dtype=float), rcond=None)
        return coeffs

    # --- log_poly_exp: T = T_ref * exp(sum_{n=1}^{N} a_n * ln(nu/nu_ref)^n) ---
    # coeffs[i] corresponds to a_{i+1} (no a_0 term).

    def log_poly_exp(self, nu: npt.ArrayLike, coeffs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        nu = np.asarray(nu, dtype=float)
        x = np.log(nu / self.nu_ref)
        coeffs = np.asarray(coeffs, dtype=float)
        exponent = sum(a * x ** (n + 1) for n, a in enumerate(coeffs))
        return self.T_ref * np.exp(exponent)

    def fit_log_poly_exp(self, nu: npt.ArrayLike, T: npt.ArrayLike, N: int) -> npt.NDArray[np.float64]:
        nu = np.asarray(nu, dtype=float)
        x = np.log(nu / self.nu_ref)
        y = np.log(np.asarray(T, dtype=float) / self.T_ref)
        design = np.vstack([x**n for n in range(1, N + 1)]).T
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        return coeffs
