import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares, minimize
import emcee

from forward_model.foregrounds.tools.FitFuncs import FitFuncs as _FG


class BayesianFitter:
    """Bayesian joint inference of foreground + Gaussian absorption signal.

    Likelihood
    ----------
    Gaussian per-channel noise:
        log L = -0.5 * sum((T_data - T_model)^2 / sigma_noise^2)

    ``sigma_noise`` can be a scalar [K] or a 1-D array [K] matching ``nu``.
    Pass the radiometer noise array from ``doRecovery.py`` for a physically
    motivated likelihood; pass a small scalar as a regularisation floor
    when no observational noise has been added.

    Foreground models
    -----------------
    'log_poly'  params = [a_0 … a_N,  A_mk, ν₀, σ]   (N+4 total)
    'sync_ff'   params = [A_s, β, T_e, EM, A_mk, ν₀, σ]   (7 total)

    Signal parameters have uniform priors with physical bounds:
        A_mk ∈ [0, ∞)    [mK]
        ν₀   ∈ [ν_min, ν_max]   [MHz]
        σ    ∈ [2, 40]   [MHz]

    Foreground priors are uniform (log_poly) or uniform-in-bounds (sync_ff).

    Parameters
    ----------
    nu       : 1-D array [MHz]
    T_data   : 1-D array [K]  — pixel-averaged sky spectrum (fg + signal)
    nu_ref   : float [MHz]
    T_ref    : float [K]
    sigma_eff: float [K]  — effective noise (default 0.01 K = 10 mK)
    """

    _SIGMA_SIG_MIN = 2.0    # MHz
    _SIGMA_SIG_MAX = 40.0   # MHz

    # sync_ff parameter bounds [lo, hi]
    _SF_LO = np.array([0.0,  1.0,  100.0,  0.0])
    _SF_HI = np.array([np.inf, 5.0, 1e5,  1e4])

    def __init__(self, nu: npt.ArrayLike, T_data: npt.ArrayLike,
                 nu_ref: float, T_ref: float,
                 sigma_noise: float | npt.ArrayLike = 0.01):
        self.nu          = np.asarray(nu, dtype=float)
        self.T_data      = np.asarray(T_data, dtype=float)
        self.nu_ref      = nu_ref
        self.T_ref       = T_ref
        self.sigma_noise = np.asarray(sigma_noise, dtype=float)   # scalar or (nfreq,)
        self._fg         = _FG(nu_ref, T_ref)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, foreground: str, N: int = 4,
            T_fg_ref: npt.ArrayLike | None = None,
            n_walkers: int = 32, n_steps: int = 1000,
            n_burn: int = 200, progress: bool = False) -> dict:
        """Run emcee and return posterior summary.

        Parameters
        ----------
        foreground : 'log_poly' or 'sync_ff'
        N          : polynomial order for log_poly (ignored for sync_ff)
        T_fg_ref   : optional clean foreground spectrum [K].
                     If given, foreground is fit to T_fg_ref (oracle mode)
                     and signals are recovered from T_data − T_fg_fit.
                     If None, the full combined spectrum is used.
        n_walkers  : number of emcee walkers (default 32)
        n_steps    : total MCMC steps per walker (default 1000)
        n_burn     : burn-in steps to discard (default 200)
        progress   : show tqdm progress bar (requires tqdm)

        Returns
        -------
        dict: A_mk_mean, A_mk_std, nu0_mean, nu0_std, sigma_mean, sigma_std,
              fg_params_mean, samples (flat chain, post burn-in), sampler
        """
        if foreground == "log_poly":
            return self._run_log_poly(N, T_fg_ref, n_walkers, n_steps, n_burn, progress)
        elif foreground == "sync_ff":
            return self._run_sync_ff(T_fg_ref, n_walkers, n_steps, n_burn, progress)
        else:
            raise ValueError(f"foreground must be 'log_poly' or 'sync_ff', got {foreground!r}")

    def fit_map(self, foreground: str, N: int = 4,
               T_fg_ref: npt.ArrayLike | None = None) -> dict:
        """MAP point estimate via L-BFGS-B. ~10–50x faster than MCMC.

        Parameters mirror fit(). Returns dict:
          A_mk_map, nu0_map, sigma_map, fg_params_map, residuals_rms, success
        """
        if foreground == "log_poly":
            return self._map_log_poly(N, T_fg_ref)
        elif foreground == "sync_ff":
            return self._map_sync_ff(T_fg_ref)
        else:
            raise ValueError(f"foreground must be 'log_poly' or 'sync_ff', got {foreground!r}")

    # ------------------------------------------------------------------
    # MAP log_poly mode
    # ------------------------------------------------------------------

    def _map_log_poly(self, N, T_fg_ref):
        T_ref_data = (np.asarray(T_fg_ref) if T_fg_ref is not None else self.T_data)

        def model(params):
            fg_coeffs        = params[:N + 1]
            A_mk, nu0, sigma = params[-3:]
            T_fg  = self._fg.log_poly(self.nu, fg_coeffs)
            T_sig = -(A_mk * 1e-3) * np.exp(-0.5 * ((self.nu - nu0) / sigma) ** 2)
            return T_fg + T_sig

        def neg_log_posterior(params):
            try:
                T_model = model(params)
                nll = 0.5 * np.sum((self.T_data - T_model) ** 2 / self.sigma_noise ** 2)
                if not np.isfinite(nll):
                    return 1e30
                return nll
            except Exception:
                return 1e30

        # Warm start: NLS foreground fit + signal grid search on residual.
        # When an oracle foreground T_fg_ref is available, subtract it directly
        # to get a residual dominated by signal rather than foreground model error.
        fg0    = self._fg.fit_log_poly(self.nu, T_ref_data, N)
        T_fg0  = self._fg.log_poly(self.nu, fg0)
        if T_fg_ref is not None:
            resid0 = self.T_data - np.asarray(T_fg_ref)
        else:
            resid0 = self.T_data - T_fg0
        A0, nu0_0, sigma_0 = self._signal_grid_search(resid0)

        x0 = np.concatenate([fg0, [A0, nu0_0, sigma_0]])

        nu_min, nu_max = float(self.nu.min()), float(self.nu.max())
        bounds = ([(None, None)] * (N + 1)
                  + [(0, None), (nu_min, nu_max),
                     (self._SIGMA_SIG_MIN, self._SIGMA_SIG_MAX)])

        res = minimize(neg_log_posterior, x0, method='L-BFGS-B', bounds=bounds)

        params_map    = res.x
        fg_params_map = params_map[:N + 1]
        A_mk_map, nu0_map, sigma_map = params_map[-3:]

        T_model_map   = model(params_map)
        residuals_rms = float(np.sqrt(np.mean((self.T_data - T_model_map) ** 2)))

        det = self._detection_stats(float(A_mk_map), float(nu0_map), float(sigma_map))
        return dict(
            A_mk_map      = float(A_mk_map),
            nu0_map       = float(nu0_map),
            sigma_map     = float(sigma_map),
            fg_params_map = fg_params_map,
            residuals_rms = residuals_rms,
            success       = bool(res.success),
            **det,
        )

    # ------------------------------------------------------------------
    # MAP sync_ff mode
    # ------------------------------------------------------------------

    def _map_sync_ff(self, T_fg_ref):
        T_ref_data = (np.asarray(T_fg_ref) if T_fg_ref is not None else self.T_data)

        def model(params):
            A_mk, nu0, sigma = params[4], params[5], params[6]
            T_fg  = self._fg.sync_ff(self.nu, params[:4])
            T_sig = -(A_mk * 1e-3) * np.exp(-0.5 * ((self.nu - nu0) / sigma) ** 2)
            return T_fg + T_sig

        def neg_log_posterior(params):
            try:
                T_model = model(params)
                nll = 0.5 * np.sum((self.T_data - T_model) ** 2 / self.sigma_noise ** 2)
                if not np.isfinite(nll):
                    return 1e30
                return nll
            except Exception:
                return 1e30

        # Warm start: NLS foreground fit + signal grid search on residual.
        # When an oracle foreground T_fg_ref is available, subtract it directly
        # to get a residual dominated by signal rather than foreground model error.
        fg0    = self._fg.fit_sync_ff(self.nu, T_ref_data)
        T_fg0  = self._fg.sync_ff(self.nu, fg0)
        if T_fg_ref is not None:
            resid0 = self.T_data - np.asarray(T_fg_ref)
        else:
            resid0 = self.T_data - T_fg0
        A0, nu0_0, sigma_0 = self._signal_grid_search(resid0)

        x0 = np.concatenate([fg0, [A0, nu0_0, sigma_0]])

        nu_min, nu_max = float(self.nu.min()), float(self.nu.max())
        bounds = [
            (0, None),                                        # A_s
            (1, 5),                                           # beta
            (100, 1e5),                                       # T_e
            (0, 1e4),                                         # EM
            (0, None),                                        # A_mk
            (nu_min, nu_max),                                 # nu0
            (self._SIGMA_SIG_MIN, self._SIGMA_SIG_MAX),       # sigma
        ]

        res = minimize(neg_log_posterior, x0, method='L-BFGS-B', bounds=bounds)

        params_map    = res.x
        fg_params_map = params_map[:4]
        A_mk_map, nu0_map, sigma_map = params_map[4], params_map[5], params_map[6]

        T_model_map   = model(params_map)
        residuals_rms = float(np.sqrt(np.mean((self.T_data - T_model_map) ** 2)))

        det = self._detection_stats(float(A_mk_map), float(nu0_map), float(sigma_map))
        return dict(
            A_mk_map      = float(A_mk_map),
            nu0_map       = float(nu0_map),
            sigma_map     = float(sigma_map),
            fg_params_map = fg_params_map,
            residuals_rms = residuals_rms,
            success       = bool(res.success),
            **det,
        )

    # ------------------------------------------------------------------
    # log_poly mode
    # ------------------------------------------------------------------

    def _run_log_poly(self, N, T_fg_ref, n_walkers, n_steps, n_burn, progress):
        # Determine data the foreground is fit against
        T_ref_data = (np.asarray(T_fg_ref) if T_fg_ref is not None else self.T_data)

        def model(params):
            fg_coeffs         = params[:N + 1]
            A_mk, nu0, sigma  = params[-3:]
            T_fg  = self._fg.log_poly(self.nu, fg_coeffs)
            T_sig = -(A_mk * 1e-3) * np.exp(-0.5 * ((self.nu - nu0) / sigma) ** 2)
            return T_fg + T_sig

        def log_prior(params):
            A_mk, nu0, sigma = params[-3:]
            if A_mk < 0:
                return -np.inf
            if not (self.nu.min() <= nu0 <= self.nu.max()):
                return -np.inf
            if not (self._SIGMA_SIG_MIN <= sigma <= self._SIGMA_SIG_MAX):
                return -np.inf
            return 0.0   # uniform on fg coefficients

        # Warm start: NLS foreground fit + signal grid search on residual.
        # When an oracle foreground T_fg_ref is available, subtract it directly
        # to get a residual dominated by signal rather than foreground model error.
        fg0    = self._fg.fit_log_poly(self.nu, T_ref_data, N)
        T_fg0  = self._fg.log_poly(self.nu, fg0)
        if T_fg_ref is not None:
            resid0 = self.T_data - np.asarray(T_fg_ref)
        else:
            resid0 = self.T_data - T_fg0
        A0, nu0_0, sigma_0 = self._signal_grid_search(resid0)

        x0 = np.concatenate([fg0, [A0, nu0_0, sigma_0]])
        ndim = len(x0)

        return self._run_mcmc(model, log_prior, x0, ndim,
                              n_walkers, n_steps, n_burn, progress,
                              signal_idx=slice(N + 1, N + 4))

    # ------------------------------------------------------------------
    # sync_ff mode
    # ------------------------------------------------------------------

    def _run_sync_ff(self, T_fg_ref, n_walkers, n_steps, n_burn, progress):
        T_ref_data = (np.asarray(T_fg_ref) if T_fg_ref is not None else self.T_data)

        def model(params):
            A_mk, nu0, sigma = params[4], params[5], params[6]
            T_fg  = self._fg.sync_ff(self.nu, params[:4])
            T_sig = -(A_mk * 1e-3) * np.exp(-0.5 * ((self.nu - nu0) / sigma) ** 2)
            return T_fg + T_sig

        def log_prior(params):
            fg_p             = params[:4]
            A_mk, nu0, sigma = params[4], params[5], params[6]
            if np.any(fg_p < self._SF_LO) or np.any(fg_p > self._SF_HI):
                return -np.inf
            if A_mk < 0:
                return -np.inf
            if not (self.nu.min() <= nu0 <= self.nu.max()):
                return -np.inf
            if not (self._SIGMA_SIG_MIN <= sigma <= self._SIGMA_SIG_MAX):
                return -np.inf
            return 0.0

        # Warm start: NLS foreground fit + signal grid search on residual.
        # When an oracle foreground T_fg_ref is available, subtract it directly
        # to get a residual dominated by signal rather than foreground model error.
        fg0    = self._fg.fit_sync_ff(self.nu, T_ref_data)
        T_fg0  = self._fg.sync_ff(self.nu, fg0)
        if T_fg_ref is not None:
            resid0 = self.T_data - np.asarray(T_fg_ref)
        else:
            resid0 = self.T_data - T_fg0
        A0, nu0_0, sigma_0 = self._signal_grid_search(resid0)

        x0   = np.concatenate([fg0, [A0, nu0_0, sigma_0]])
        ndim = 7

        return self._run_mcmc(model, log_prior, x0, ndim,
                              n_walkers, n_steps, n_burn, progress,
                              signal_idx=slice(4, 7))

    # ------------------------------------------------------------------
    # MCMC engine
    # ------------------------------------------------------------------

    def _log_likelihood(self, params, model):
        try:
            T_model = model(params)
        except Exception:
            return -np.inf
        return -0.5 * np.sum((self.T_data - T_model) ** 2 / self.sigma_noise ** 2)

    def _run_mcmc(self, model, log_prior, x0, ndim,
                  n_walkers, n_steps, n_burn, progress, signal_idx):
        def log_prob(params):
            lp = log_prior(params)
            if not np.isfinite(lp):
                return -np.inf
            return lp + self._log_likelihood(params, model)

        # Scatter walkers around the warm start (1% perturbation)
        rng = np.random.default_rng(42)
        scale = np.abs(x0) * 0.01 + 1e-4
        p0 = x0 + rng.normal(0, 1, (n_walkers, ndim)) * scale

        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
        sampler.run_mcmc(p0, n_steps, progress=progress)

        flat = sampler.get_chain(discard=n_burn, flat=True)   # (n_samples, ndim)

        sig_samples = flat[:, signal_idx]   # columns: [A_mk, nu0, sigma]
        fg_samples  = np.delete(flat, range(*signal_idx.indices(ndim)), axis=1)

        return dict(
            A_mk_mean     = float(sig_samples[:, 0].mean()),
            A_mk_std      = float(sig_samples[:, 0].std()),
            nu0_mean      = float(sig_samples[:, 1].mean()),
            nu0_std       = float(sig_samples[:, 1].std()),
            sigma_mean    = float(sig_samples[:, 2].mean()),
            sigma_std     = float(sig_samples[:, 2].std()),
            fg_params_mean= fg_samples.mean(axis=0),
            samples       = flat,
            sampler       = sampler,
        )

    # ------------------------------------------------------------------
    # Signal warm-start: grid search
    # ------------------------------------------------------------------

    def _signal_grid_search(self, resid: npt.NDArray,
                            n_nu0: int = 40, n_sigma: int = 20) -> tuple[float, float, float]:
        nu0_vals   = np.linspace(self.nu.min(), self.nu.max(), n_nu0)
        sigma_vals = np.linspace(self._SIGMA_SIG_MIN, self._SIGMA_SIG_MAX, n_sigma)

        best_sse = np.inf
        best = (max(-float(resid.min()) * 1e3, 1.0),
                float(self.nu[np.argmin(resid)]),
                10.0)

        for nu0 in nu0_vals:
            for sigma in sigma_vals:
                template = np.exp(-0.5 * ((self.nu - nu0) / sigma) ** 2)
                denom    = np.dot(template, template)
                if denom < 1e-30:
                    continue
                A_mk = max(-1e3 * np.dot(resid, template) / denom, 0.0)
                T_sig = -(A_mk * 1e-3) * template
                sse   = float(np.sum((resid - T_sig) ** 2))
                if sse < best_sse:
                    best_sse = sse
                    best     = (A_mk, float(nu0), float(sigma))
        return best

    # ------------------------------------------------------------------
    # Detection statistics (matched-filter Cramér-Rao bound on A_mk)
    # ------------------------------------------------------------------

    def _detection_stats(self, A_mk_map: float, nu0_map: float,
                         sigma_map: float) -> dict:
        """Compute effective noise and detection SNR for the MAP amplitude.

        Uses the matched-filter (Fisher information) bound:
            sigma_A_eff [mK] = 1e3 / sqrt(sum(template^2 / sigma_noise_K^2))
        where template = exp(-0.5 * ((nu - nu0) / sigma)^2).

        Returns sigma_A_eff [mK] and detection_snr = A_mk_map / sigma_A_eff.
        """
        template    = np.exp(-0.5 * ((self.nu - nu0_map) / sigma_map) ** 2)
        fisher_A    = np.sum(template ** 2 / self.sigma_noise ** 2)   # [K^-2]
        sigma_A_eff = 1e3 / np.sqrt(fisher_A) if fisher_A > 0 else np.inf  # [mK]
        detection_snr = A_mk_map / sigma_A_eff if sigma_A_eff > 0 else 0.0
        return dict(sigma_A_eff=float(sigma_A_eff),
                    detection_snr=float(detection_snr))
