import numpy as np
import numpy.typing as npt
from itertools import product
from scipy.optimize import least_squares

from forward_model.foregrounds.tools.FitFuncs import FitFuncs as _ForegroundFitFuncs


class FitFuncs(_ForegroundFitFuncs):
    """foregrounds.tools.FitFuncs.FitFuncs (foreground spectral models)
    plus 21-cm signal templates and joint foreground+signal fitting."""

    # --- 21-cm signal templates ---

    @staticmethod
    def gaussian_21cm(nu: npt.ArrayLike, A: float, nu0: float, w: float) -> npt.NDArray[np.float64]:
        """Plain Gaussian 21-cm template, FWHM-parameterized (w = full
        width at half maximum, same convention as flattened_gaussian_21cm)."""
        nu = np.asarray(nu, dtype=float)
        return A * np.exp(-4 * np.log(2) * (nu - nu0) ** 2 / w ** 2)

    @staticmethod
    def flattened_gaussian_21cm(nu: npt.ArrayLike, A: float, nu0: float, w: float,
                                 tau: float) -> npt.NDArray[np.float64]:
        """EDGES-style flattened Gaussian absorption template (Bowman et
        al. 2018, Nature). Reduces to gaussian_21cm as tau -> 0. tau must
        be > 0 (division by zero at tau=0); large-magnitude or negative
        tau drives the formula into a numerically unstable regime
        (exp overflow) -- raises for tau outside (0, 50]."""
        if not (0 < tau <= 50):
            raise ValueError(f"tau must be in (0, 50] (numerically unstable outside this range), got {tau}")
        nu = np.asarray(nu, dtype=float)
        B = (4 * (nu - nu0) ** 2 / w ** 2) * np.log(-(1.0 / tau) * np.log((1 + np.exp(-tau)) / 2))
        return A * (1 - np.exp(-tau * np.exp(B))) / (1 - np.exp(-tau))

    def _signal_shape(self, nu, signal: str, shape_params) -> npt.NDArray[np.float64]:
        """Unit-amplitude (A=1) signal template evaluated at nu."""
        if signal == "gaussian":
            nu0, w = shape_params
            return self.gaussian_21cm(nu, 1.0, nu0, w)
        elif signal == "flattened_gaussian":
            nu0, w, tau = shape_params
            return self.flattened_gaussian_21cm(nu, 1.0, nu0, w, tau)
        raise ValueError(f"signal must be 'gaussian' or 'flattened_gaussian', got {signal!r}")

    def _linear_subproblem(self, nu, T, foreground: str, signal: str, N: int, shape_params):
        """For log_poly/power_poly: exact linear lstsq over [foreground
        coeffs, A] at fixed signal shape params. Returns (residual,
        fg_coeffs, A)."""
        shape_col = self._signal_shape(nu, signal, shape_params)
        design = np.vstack(self._foreground_design_columns(nu, foreground, N) + [shape_col]).T
        coeffs, *_ = np.linalg.lstsq(design, T, rcond=None)
        return T - design @ coeffs, coeffs[:-1], coeffs[-1]

    def _log_poly_exp_subproblem(self, nu, T, signal: str, fg_coeffs, shape_params):
        """For log_poly_exp: given its own (nonlinear) coeffs and signal
        shape params, solve amplitude A by simple linear regression.
        Returns (residual, A)."""
        fg_pred = self.log_poly_exp(nu, fg_coeffs)
        shape_col = self._signal_shape(nu, signal, shape_params)
        resid = T - fg_pred
        denom = np.dot(shape_col, shape_col)
        A = np.dot(shape_col, resid) / denom if denom > 0 else 0.0
        return T - fg_pred - A * shape_col, A

    def fit_combined(self, nu: npt.ArrayLike, T: npt.ArrayLike, foreground: str, signal: str,
                      N: int, n_grid: int = 8) -> dict:
        """Jointly fit a foreground model plus an additive 21-cm signal
        template. foreground in {"log_poly","power_poly","log_poly_exp"},
        signal in {"gaussian","flattened_gaussian"}.

        Uses a separable (VARPRO-style) approach: a coarse grid search
        over the signal's nonlinear shape parameters (nu0, w, [tau]) --
        for log_poly/power_poly the foreground coefficients and signal
        amplitude A are solved exactly via linear lstsq at each grid
        point; for log_poly_exp (whose own coefficients are nonlinear
        once a signal is added additively) a local refinement over its
        coefficients is done at each grid point too, seeded from its
        existing (foreground-only) linear log-space fit. The best grid
        point is then locally polished.

        Refinement uses scipy.optimize.least_squares (Levenberg-Marquardt
        for unbounded sub-problems, trust-region-reflective where bounds
        apply) rather than a generic derivative-free minimizer -- this
        problem is a smooth sum-of-squared-residuals objective, exactly
        what least_squares is designed for, and converges far more
        reliably here than Nelder-Mead (empirically confirmed: Nelder-Mead
        plateaued at SSE values ~2-2.5x worse than least_squares achieves
        from the same starting point, regardless of iteration budget).

        Returns a dict with keys: "foreground_coeffs", "A", "nu0", "w",
        (and "tau" if signal="flattened_gaussian"), "sse".
        """
        nu = np.asarray(nu, dtype=float)
        T = np.asarray(T, dtype=float)

        nu0_grid = np.linspace(nu.min(), nu.max(), n_grid)
        bandwidth = nu.max() - nu.min()
        w_grid = np.linspace(bandwidth / 10, bandwidth, n_grid)
        shape_lo, shape_hi = [nu.min(), bandwidth / 10], [nu.max(), bandwidth]
        if signal == "flattened_gaussian":
            tau_grid = np.geomspace(0.1, 20, n_grid)
            grid_points = list(product(nu0_grid, w_grid, tau_grid))
            shape_lo.append(0.1)
            shape_hi.append(20)
        else:
            grid_points = list(product(nu0_grid, w_grid))

        if foreground == "log_poly_exp":
            fg_coeffs_guess = self.fit_log_poly_exp(nu, T, N)

            def evaluate(shape_params, fg_x0):
                res = least_squares(
                    lambda fg: self._log_poly_exp_subproblem(nu, T, signal, fg, shape_params)[0],
                    fg_x0, method="lm",
                )
                resid, A = self._log_poly_exp_subproblem(nu, T, signal, res.x, shape_params)
                return np.sum(resid**2), res.x, A

            best_sse, best_shape, best_fg, best_A = np.inf, None, None, None
            for shape_params in grid_points:
                sse, fg_coeffs, A = evaluate(shape_params, fg_coeffs_guess)
                if sse < best_sse:
                    best_sse, best_shape, best_fg, best_A = sse, shape_params, fg_coeffs, A

            def full_residual(x):
                fg = x[:N]
                shape_params = x[N:]
                return self._log_poly_exp_subproblem(nu, T, signal, fg, shape_params)[0]

            x0 = np.concatenate([best_fg, best_shape])
            lo = [-np.inf] * N + shape_lo
            hi = [np.inf] * N + shape_hi
            res = least_squares(full_residual, x0, method="trf", bounds=(lo, hi))
            fg_coeffs = res.x[:N]
            shape_params = res.x[N:]
            resid, A = self._log_poly_exp_subproblem(nu, T, signal, fg_coeffs, shape_params)
            sse = np.sum(resid**2)
        else:
            best_sse, best_shape = np.inf, None
            for shape_params in grid_points:
                resid, _, _ = self._linear_subproblem(nu, T, foreground, signal, N, shape_params)
                sse = np.sum(resid**2)
                if sse < best_sse:
                    best_sse, best_shape = sse, shape_params

            def residual_fn(shape_params):
                return self._linear_subproblem(nu, T, foreground, signal, N, shape_params)[0]

            res = least_squares(residual_fn, best_shape, method="trf", bounds=(shape_lo, shape_hi))
            shape_params = res.x
            resid, fg_coeffs, A = self._linear_subproblem(nu, T, foreground, signal, N, shape_params)
            sse = np.sum(resid**2)

        result = {"foreground_coeffs": fg_coeffs, "A": A, "nu0": shape_params[0], "w": shape_params[1], "sse": sse}
        if signal == "flattened_gaussian":
            result["tau"] = shape_params[2]
        return result
