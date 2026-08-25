"""Convolve 4 beams of increasing complexity (two single-mode TM-only
beams, and two mixed beams drawing terms up to l=20) with the GSM2008
sky over 40-130 MHz, fit each resulting T_antenna(freq) spectrum with
FitFuncs.log_poly_exp, and plot all 4 beams' fit residuals overlaid.
"""
import time
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from tools.Sky import Sky
from tools.VshBasis import VshBasis
from tools.EField import EField
from tools.FitFuncs import FitFuncs
from tools.Convolution import beam_power_coefficients, sky_coefficients, antenna_temperature
from tools.config import PROJECT_ROOT, OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

FREQ_MIN, FREQ_MAX = 40.0, 130.0
N_POINTS = 50
HEIGHT = 1.0  # m
L_MAX = 40
NSIDE = 128
FIT_N = 5

all_lm_mode_wide = [(l, m, mode) for l in range(1, 21) for m in range(-l, l + 1) for mode in ("TM", "TE")]


def random_mix(seed: int, n_terms: int) -> VshBasis:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_lm_mode_wide), size=n_terms, replace=False)
    terms = [all_lm_mode_wide[i] for i in idx]
    coeffs = rng.uniform(-1.0, 1.0, size=n_terms)
    return VshBasis(terms, coeffs)


beams = [
    ("l=1, m=-1", VshBasis([(1, -1, "TM")], [1.0])),
    ("l=2, m=0", VshBasis([(2, 0, "TM")], [1.0])),
    ("6-term mix (l<=20)", random_mix(1, 6)),
    ("12-term mix (l<=20)", random_mix(2, 12)),
]

sky = Sky(model="2008")
freqs_mhz = np.linspace(FREQ_MIN, FREQ_MAX, N_POINTS)
nu_ref = np.sqrt(FREQ_MIN * FREQ_MAX)

print(f"Building design matrix at l_max={L_MAX}, nside={NSIDE} (one-time cost)...")
t0 = time.perf_counter()
sky_maps = sky.generate(freqs_mhz)
sky_coeffs_per_freq = [sky_coefficients(sky_maps[i], l_max=L_MAX, nside=NSIDE) for i in range(N_POINTS)]
print(f"Sky generation + projection for {N_POINTS} frequencies: {time.perf_counter() - t0:.2f} s")

sky_map_ref = sky.generate(nu_ref)
sky_coeffs_ref = sky_coefficients(sky_map_ref, l_max=L_MAX, nside=NSIDE)

results = {}
for label, basis in beams:
    t0 = time.perf_counter()
    T_values = np.empty(N_POINTS)
    for i, freq_mhz in enumerate(freqs_mhz):
        ef = EField.from_basis(basis, freq_mhz * 1e6, HEIGHT)
        beam_coeffs = beam_power_coefficients(ef, l_max=L_MAX, nside=NSIDE)
        T_values[i] = antenna_temperature(beam_coeffs, sky_coeffs_per_freq[i])

    ef_ref = EField.from_basis(basis, nu_ref * 1e6, HEIGHT)
    beam_coeffs_ref = beam_power_coefficients(ef_ref, l_max=L_MAX, nside=NSIDE)
    T_ref = antenna_temperature(beam_coeffs_ref, sky_coeffs_ref)

    elapsed = time.perf_counter() - t0
    results[label] = (T_values, T_ref)
    print(f"{label}: {elapsed:.2f} s for {N_POINTS} frequencies + reference point")

# Lightweight correctness check at the new (l_max, nside) resolution: brute-force
# pixel-space cross-check for the first beam at the reference frequency.
label0, basis0 = beams[0]
ef0 = EField.from_basis(basis0, nu_ref * 1e6, HEIGHT)
npix = hp.nside2npix(NSIDE)
theta, phi = hp.pix2ang(NSIDE, np.arange(npix))
pixel_area = 4 * np.pi / npix
e_theta, e_phi = ef0.apply_image_theory(theta, phi)
beam_power = e_theta**2 + e_phi**2
sky_interp = hp.get_interp_val(sky_map_ref, theta, phi)
T_brute = np.sum(beam_power * sky_interp * pixel_area) / np.sum(beam_power * pixel_area)
T_fast = results[label0][1]
rel_diff = abs(T_fast - T_brute) / abs(T_brute)

print(f"\nBrute-force cross-check ({label0}, f={nu_ref:.2f} MHz, l_max={L_MAX}, nside={NSIDE}):")
print(f"  fast (Parseval):  T_antenna = {T_fast:.4f} K")
print(f"  brute force:      T_antenna = {T_brute:.4f} K")
print(f"  relative diff:    {rel_diff:.4g}")

# Fit each beam's spectrum with log_poly_exp and compute residuals
fig, ax = plt.subplots(figsize=(10, 6))
linestyles = ["-", "--", "-.", ":"]
print("\nFit results:")
for (label, (T_values, T_ref)), linestyle in zip(results.items(), linestyles):
    ff = FitFuncs(nu_ref, T_ref)
    coeffs = ff.fit_log_poly_exp(freqs_mhz, T_values, FIT_N)
    T_fit = ff.log_poly_exp(freqs_mhz, coeffs)
    resid = T_values - T_fit
    rms = np.sqrt(np.mean(resid**2))
    print(f"  {label}: RMS residual = {rms:.4g} K")
    ax.plot(freqs_mhz, resid, label=label, linestyle=linestyle)

ax.axhline(0, color="gray", lw=0.8)
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Residual (K)")
ax.set_title(f"log_poly_exp fit residuals (N={FIT_N}) vs beam complexity, GSM2008, h={HEIGHT} m")
ax.legend()
ax.grid(True)
ax.set_xlim(FREQ_MIN, FREQ_MAX)
fig.tight_layout()

save_dir = OUTPUT_DIR / "tests"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "Beam_scan_foreground_fit.png"
fig.savefig(save_path)
print(f"\nSaved to {save_path}")
