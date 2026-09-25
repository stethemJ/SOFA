"""Fit three foreground spectral models (FitFuncs.py) to the spatially
averaged antenna temperature from the spectral-index sky model over
40-130 MHz. All three models are given 5 free parameters for comparison.
"""
import numpy as np
import matplotlib.pyplot as plt

from forward_model.foregrounds.tools.Sky import Sky
from forward_model.foregrounds.tools.FitFuncs import FitFuncs
from forward_model.tools.config import PROJECT_ROOT
from forward_model.foregrounds.tools.config import OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

FREQ_MIN, FREQ_MAX = 40.0, 130.0
N_POINTS = 100
N_ORDER = 4  # log_poly: N+1=5 coeffs; power_poly/log_poly_exp: N=5 coeffs

nu = np.linspace(FREQ_MIN, FREQ_MAX, N_POINTS)
sky = Sky()
T_data = sky.generate(nu).mean(axis=1)

nu_ref = np.sqrt(FREQ_MIN * FREQ_MAX)
T_ref = float(sky.generate(nu_ref).mean())

ff = FitFuncs(nu_ref, T_ref)

fit_models = [
    ("log_poly", ff.log_poly, ff.fit_log_poly, N_ORDER),
    ("power_poly", ff.power_poly, ff.fit_power_poly, N_ORDER + 1),
    ("log_poly_exp", ff.log_poly_exp, ff.fit_log_poly_exp, N_ORDER + 1),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex="col", height_ratios=[3, 1])

for col, (name, model_fn, fit_fn, N) in enumerate(fit_models):
    coeffs = fit_fn(nu, T_data, N)
    T_fit = model_fn(nu, coeffs)
    resid = T_data - T_fit
    rms = np.sqrt(np.mean(resid**2))
    print(f"{name}: {len(coeffs)} params, RMS residual = {rms:.4g} K")

    ax_fit = axes[0, col]
    ax_fit.plot(nu, T_data, label="spectral-index sky mean", color="black", lw=1)
    ax_fit.plot(nu, T_fit, label="fit", linestyle="--")
    ax_fit.set_title(f"{name} (N={N}, RMS={rms:.3g} K)")
    ax_fit.set_ylabel("Mean Sky Temperature (K)")
    ax_fit.legend()
    ax_fit.grid(True)

    ax_res = axes[1, col]
    ax_res.plot(nu, resid, color="firebrick")
    ax_res.axhline(0, color="gray", lw=0.8)
    ax_res.set_xlabel("Frequency (MHz)")
    ax_res.set_ylabel("Residual (K)")
    ax_res.grid(True)

fig.suptitle(f"Foreground model fits, spectral-index sky, {FREQ_MIN:.0f}-{FREQ_MAX:.0f} MHz (nu_ref={nu_ref:.1f} MHz)")
fig.tight_layout()

save_path = OUTPUT_DIR / "sky_model_fit.png"
fig.savefig(save_path)
print(f"Saved to {save_path}")
