"""Fit log-space polynomial and synchrotron+free-free models to the
spatially averaged antenna temperature from the spectral-index sky model
over 40–130 MHz."""
import numpy as np
import matplotlib.pyplot as plt

from forward_model.foregrounds.tools.Sky import Sky
from forward_model.foregrounds.tools.FitFuncs import FitFuncs
from forward_model.tools.config import PROJECT_ROOT
from forward_model.foregrounds.tools.config import OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

FREQ_MIN, FREQ_MAX = 40.0, 130.0
N_POINTS = 100
LOG_POLY_ORDER = 4  # N+1 = 5 free parameters

nu = np.linspace(FREQ_MIN, FREQ_MAX, N_POINTS)
sky = Sky()
T_data = sky.generate(nu).mean(axis=1)

nu_ref = np.sqrt(FREQ_MIN * FREQ_MAX)
T_ref = float(sky.generate(nu_ref).mean())

ff = FitFuncs(nu_ref, T_ref)

# --- log_poly fit ---
lp_coeffs = ff.fit_log_poly(nu, T_data, LOG_POLY_ORDER)
T_lp = ff.log_poly(nu, lp_coeffs)
resid_lp = T_data - T_lp
rms_lp = np.sqrt(np.mean(resid_lp**2))
print(f"log_poly (N={LOG_POLY_ORDER}): {len(lp_coeffs)} params, RMS = {rms_lp:.4g} K")

# --- sync_ff fit ---
sf_params = ff.fit_sync_ff(nu, T_data)
T_sf = ff.sync_ff(nu, sf_params)
resid_sf = T_data - T_sf
rms_sf = np.sqrt(np.mean(resid_sf**2))
A_s, beta, te, em = sf_params
print(f"sync_ff: 4 params, RMS = {rms_sf:.4g} K")
print(f"  A_s={A_s:.4g} K  beta={beta:.4f}  T_e={te:.1f} K  EM={em:.4g} pc/cm^6")

# --- plot ---
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col", height_ratios=[3, 1])

for col, (name, T_fit, resid, rms, label) in enumerate([
    ("log_poly", T_lp, resid_lp, rms_lp,
     f"log-poly N={LOG_POLY_ORDER} ({LOG_POLY_ORDER+1} params)"),
    ("sync_ff", T_sf, resid_sf, rms_sf,
     rf"sync+ff: $A_s$={A_s:.3g} K, $\beta$={beta:.3f}, $T_e$={te:.0f} K, EM={em:.3g}"),
]):
    ax_fit = axes[0, col]
    ax_fit.plot(nu, T_data, label="spectral-index sky mean", color="black", lw=1.5)
    ax_fit.plot(nu, T_fit, label="fit", linestyle="--", lw=1.5)
    ax_fit.set_title(f"{name}  (RMS = {rms:.3g} K)")
    ax_fit.set_ylabel("Mean Sky Temperature (K)")
    ax_fit.legend(fontsize=8)
    ax_fit.grid(True)

    ax_res = axes[1, col]
    ax_res.plot(nu, resid, color="firebrick", lw=1)
    ax_res.axhline(0, color="gray", lw=0.8)
    ax_res.set_xlabel("Frequency (MHz)")
    ax_res.set_ylabel("Residual (K)")
    ax_res.grid(True)

    ax_fit.annotate(label, xy=(0.03, 0.05), xycoords="axes fraction", fontsize=7,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

fig.suptitle(
    f"Foreground model fits, spectral-index sky, "
    f"{FREQ_MIN:.0f}–{FREQ_MAX:.0f} MHz  (nu_ref = {nu_ref:.1f} MHz)"
)
fig.tight_layout()

save_path = OUTPUT_DIR / "sky_model_fit.png"
fig.savefig(save_path)
print(f"Saved to {save_path}")
