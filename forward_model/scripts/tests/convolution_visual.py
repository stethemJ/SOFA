"""Sweep the fast beam-sky convolution (tools/Convolution.py) over
40-130 MHz for beams with different numbers of VSH modes, against the
GSM2008 sky, and plot the resulting antenna-temperature spectra
overlaid. Includes a brute-force pixel-space cross-check at one
reference frequency to confirm the fast (design-matrix-cached) method
is correct.
"""
import time
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from forward_model.foregrounds.tools.Sky import Sky
from antenna.tools.VshBasis import VshBasis
from antenna.tools.EField import EField
from forward_model.tools.Convolution import (
    beam_power_coefficients, sky_coefficients, antenna_temperature, DEFAULT_BEAM_NSIDE,
)
from forward_model.tools.config import PROJECT_ROOT
from antenna.tools.config import OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

FREQ_MIN, FREQ_MAX = 40.0, 130.0
N_POINTS = 50
HEIGHT = 1.0  # m
REFERENCE_FREQ_MHZ = 100.0  # for the brute-force cross-check

all_lm_mode = [(l, m, mode) for l in range(1, 5) for m in range(-l, l + 1) for mode in ("TM", "TE")]


def random_mix(seed: int, n_terms: int) -> VshBasis:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_lm_mode), size=n_terms, replace=False)
    terms = [all_lm_mode[i] for i in idx]
    coeffs = rng.uniform(-1.0, 1.0, size=n_terms)
    return VshBasis(terms, coeffs)


beams = [
    ("Single dipole (1 term)", VshBasis([(1, 1, "TM")], [1.0])),
    ("Medium mix (5 terms)", random_mix(1, 5)),
    ("Large mix (12 terms)", random_mix(2, 12)),
]

sky = Sky(nside=DEFAULT_BEAM_NSIDE)
freqs_mhz = np.linspace(FREQ_MIN, FREQ_MAX, N_POINTS)

t0 = time.perf_counter()
sky_maps = sky.generate(freqs_mhz)
sky_coeffs_per_freq = [sky_coefficients(sky_maps[i]) for i in range(N_POINTS)]
print(f"Sky generation + projection for {N_POINTS} frequencies: {time.perf_counter() - t0:.3f} s")

results = {}
for label, basis in beams:
    t0 = time.perf_counter()
    T_values = np.empty(N_POINTS)
    for i, freq_mhz in enumerate(freqs_mhz):
        ef = EField.from_basis(basis, freq_mhz * 1e6, HEIGHT)
        beam_coeffs = beam_power_coefficients(ef)
        T_values[i] = antenna_temperature(beam_coeffs, sky_coeffs_per_freq[i])
    elapsed = time.perf_counter() - t0
    results[label] = T_values
    print(f"{label}: {elapsed:.3f} s for {N_POINTS} frequencies")

# Brute-force pixel-space cross-check at one reference frequency (single-term dipole).
# Use the actual sampled grid frequency closest to REFERENCE_FREQ_MHZ (not a
# separately hardcoded value) so the fast and brute-force sides use identical
# inputs -- otherwise a mismatched frequency looks like a correctness error.
idx_ref = np.argmin(np.abs(freqs_mhz - REFERENCE_FREQ_MHZ))
freq_ref = freqs_mhz[idx_ref]

label0, basis0 = beams[0]
ef0 = EField.from_basis(basis0, freq_ref * 1e6, HEIGHT)
sky_map_ref = sky_maps[idx_ref]

nside = DEFAULT_BEAM_NSIDE
npix = hp.nside2npix(nside)
theta, phi = hp.pix2ang(nside, np.arange(npix))
pixel_area = 4 * np.pi / npix
e_theta, e_phi = ef0.apply_image_theory(theta, phi)
beam_power = e_theta**2 + e_phi**2
sky_interp = hp.get_interp_val(sky_map_ref, theta, phi)
T_brute = np.sum(beam_power * sky_interp * pixel_area) / np.sum(beam_power * pixel_area)

T_fast = results[label0][idx_ref]
rel_diff = abs(T_fast - T_brute) / abs(T_brute)

print(f"\nBrute-force cross-check ({label0}, f={freq_ref:.2f} MHz):")
print(f"  fast (Parseval):  T_antenna = {T_fast:.4f} K")
print(f"  brute force:      T_antenna = {T_brute:.4f} K")
print(f"  relative diff:    {rel_diff:.4g}")

# Plot: T_antenna(freq) spectrum, all beams overlaid
fig, ax = plt.subplots(figsize=(10, 6))
for label, T_values in results.items():
    ax.plot(freqs_mhz, T_values, label=label)

ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Antenna Temperature (K)")
ax.set_title(f"Beam-convolved antenna temperature vs frequency (GSM2008, h={HEIGHT} m)")
ax.legend()
ax.grid(True)
fig.tight_layout()

save_dir = OUTPUT_DIR / "tests"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "convolution_visual.png"
fig.savefig(save_path)
print(f"\nSaved to {save_path}")
