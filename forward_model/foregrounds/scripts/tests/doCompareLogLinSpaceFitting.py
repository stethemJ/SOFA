"""Compare three ways of fitting a traditional (unanchored) degree-5
log-log polynomial, ln(T) = a0 + a1*ln(nu) + ... + a5*ln(nu)^5, to the
GSM2008 foreground: (1) linear least squares in log space (exact,
closed-form, minimizes relative/log-space error), (2) nonlinear least
squares in linear space (iterative, minimizes the actual temperature
residual), and (3) weighted least squares in log space (closed-form,
weights=T**2, approximating (2)'s objective via Iteratively Reweighted
Least Squares -- see tools/FitFuncs.fit_log_log_poly_weighted). These
are different objectives, so they converge to different coefficients
-- this plots the resulting linear-space residuals for all three.
"""
import numpy as np
import matplotlib.pyplot as plt

from forward_model.foregrounds.tools.Sky import Sky
from forward_model.foregrounds.tools.FitFuncs import FitFuncs
from forward_model.tools.config import PROJECT_ROOT
from forward_model.foregrounds.tools.config import OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

FREQ_MIN, FREQ_MAX, FREQ_STEP = 20.0, 50.0, 0.5  # band of interest -- shared with foreground_visual.py / 21cm_visual.py
PAD_MHZ = 1.0  # extra margin fit on each side, to keep the reported band away from the polynomial's edge effects
N = 9

# Fit over the padded range (gives the polynomial context beyond the
# edges of the band we actually care about); report/plot only over the
# band of interest, sliced back out below.
freq_mhz = np.arange(FREQ_MIN - PAD_MHZ, FREQ_MAX + PAD_MHZ + FREQ_STEP, FREQ_STEP)
T_data = Sky().generate(freq_mhz).mean(axis=1)

ff = FitFuncs(nu_ref=np.sqrt(FREQ_MIN * FREQ_MAX), T_ref=float(np.mean(T_data)))

coeffs_lin = ff.fit_log_log_poly(freq_mhz, T_data, N)
coeffs_nl = ff.fit_log_log_poly_nonlinear(freq_mhz, T_data, N, x0=coeffs_lin)
coeffs_wls = ff.fit_log_log_poly_weighted(freq_mhz, T_data, N, n_iter=5)

band_mask = (freq_mhz >= FREQ_MIN) & (freq_mhz <= FREQ_MAX)
freq_mhz = freq_mhz[band_mask]
T_data = T_data[band_mask]

resid_lin = T_data - ff.log_log_poly(freq_mhz, coeffs_lin)
resid_nl = T_data - ff.log_log_poly(freq_mhz, coeffs_nl)
resid_wls = T_data - ff.log_log_poly(freq_mhz, coeffs_wls)

rms_lin = np.sqrt(np.mean(resid_lin**2))
rms_nl = np.sqrt(np.mean(resid_nl**2))
rms_wls = np.sqrt(np.mean(resid_wls**2))
print(f"Linear least squares (log space):       coeffs={coeffs_lin}, RMS residual={rms_lin:.4g} K")
print(f"Nonlinear least squares (linear space):  coeffs={coeffs_nl}, RMS residual={rms_nl:.4g} K")
print(f"Weighted least squares (log space, IRLS): coeffs={coeffs_wls}, RMS residual={rms_wls:.4g} K")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(freq_mhz, resid_lin * 1000, label=f"Linear least squares (log space), RMS={rms_lin:.3g} K")
ax.plot(freq_mhz, resid_nl * 1000, label=f"Nonlinear least squares (linear space), RMS={rms_nl:.3g} K", linestyle="--")
ax.plot(freq_mhz, resid_wls * 1000, label=f"Weighted least squares (log space, IRLS), RMS={rms_wls:.3g} K", linestyle=":")
ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Residual, T_data - T_fit (mK)")
ax.set_title(f"log-log polynomial (N={N}) fit residuals: GSM2008 foreground")
ax.legend()
ax.grid(True)
fig.tight_layout()

save_dir = OUTPUT_DIR / "tests"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "doCompareLogLinSpaceFitting.png"
fig.savefig(save_path)
print(f"Saved to {save_path}")
