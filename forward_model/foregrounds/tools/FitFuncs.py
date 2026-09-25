import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares


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

    # --- log_log_poly: traditional (unanchored) log-log polynomial ---
    # ln(T) = sum_{n=0}^{N} a_n * ln(nu)^n, i.e. T = exp(polyval(ln(nu))).
    # Unlike log_poly_exp, this has a free a_0 term and is not normalized
    # by nu_ref/T_ref -- provided to compare a closed-form linear fit in
    # log space against a genuinely nonlinear fit in linear space.

    def log_log_poly(self, nu: npt.ArrayLike, coeffs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        nu = np.asarray(nu, dtype=float)
        x = np.log(nu)
        coeffs = np.asarray(coeffs, dtype=float)
        exponent = sum(a * x**n for n, a in enumerate(coeffs))
        return np.exp(exponent)

    def fit_log_log_poly(self, nu: npt.ArrayLike, T: npt.ArrayLike, N: int) -> npt.NDArray[np.float64]:
        """Linear least squares in log space: exact, closed-form, but
        minimizes error in ln(T) (i.e. relative error in T), not the
        actual linear-space residual."""
        nu = np.asarray(nu, dtype=float)
        x = np.log(nu)
        y = np.log(np.asarray(T, dtype=float))
        design = np.vstack([x**n for n in range(N + 1)]).T
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        return coeffs

    def fit_log_log_poly_weighted(self, nu: npt.ArrayLike, T: npt.ArrayLike, N: int,
                                   n_iter: int = 1) -> npt.NDArray[np.float64]:
        """Weighted least squares in log space, weights = T**2. Stays
        closed-form (a linear solve) but approximates the nonlinear
        fit's actual-linear-space-residual objective: weighting by T**2
        turns the log-space normal equations' stationarity condition
        into sum[(T - T_hat) * T_hat * x**j] = 0, matching the nonlinear
        fit's own stationarity condition in the small-residual limit.
        n_iter=1 uses fixed weights from the data (a single weighted
        solve); n_iter>1 re-derives weights from the fit's own
        prediction each pass (Iteratively Reweighted Least Squares),
        converging closer to the nonlinear optimum while every
        individual step remains an exact linear solve."""
        nu = np.asarray(nu, dtype=float)
        T = np.asarray(T, dtype=float)
        x = np.log(nu)
        y = np.log(T)
        design = np.vstack([x**n for n in range(N + 1)]).T
        weights = T**2
        coeffs = None
        for _ in range(max(n_iter, 1)):
            sw = np.sqrt(weights)
            coeffs, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
            weights = self.log_log_poly(nu, coeffs) ** 2
        return coeffs

    def fit_log_log_poly_nonlinear(self, nu: npt.ArrayLike, T: npt.ArrayLike, N: int,
                                    x0: npt.ArrayLike = None) -> npt.NDArray[np.float64]:
        """Nonlinear least squares directly in linear space: minimizes
        the actual (T - model) residual, at the cost of needing an
        iterative solver. Seeds from fit_log_log_poly's result when x0
        isn't given -- a natural warm start for the same functional form."""
        nu = np.asarray(nu, dtype=float)
        T = np.asarray(T, dtype=float)
        if x0 is None:
            x0 = self.fit_log_log_poly(nu, T, N)
        res = least_squares(lambda coeffs: T - self.log_log_poly(nu, coeffs), x0, method="lm")
        return res.x

    def _foreground_design_columns(self, nu, foreground: str, N: int):
        """Linear design-matrix columns for log_poly/power_poly (not
        valid for log_poly_exp, which is nonlinear in its own coeffs)."""
        if foreground == "log_poly":
            x = np.log(nu / self.nu_ref)
            return [self.T_ref * x**n for n in range(N + 1)]
        elif foreground == "power_poly":
            return [nu ** (n - 2.5) for n in range(N)]
        raise ValueError(f"foreground must be 'log_poly' or 'power_poly' here, got {foreground!r}")
