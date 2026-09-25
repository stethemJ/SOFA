"""Plot the scaled angular power D_l = l(l+1)C_l/(2*pi) from the
spectral-index foreground sky as a heatmap over frequency (10-130 MHz)
and multipole l (1-40). Includes a healpy cross-check at one reference
frequency to confirm the fast (cached design-matrix) projection is
consistent with hp.anafast.
"""
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from forward_model.foregrounds.tools.Sky import Sky
from forward_model.tools.Convolution import sky_coefficients
from forward_model.tools.config import PROJECT_ROOT
from forward_model.foregrounds.tools.config import OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

FREQ_MIN, FREQ_MAX = 10.0, 130.0
N_POINTS = 50
L_MAX = 40
NSIDE = 128


def c_l_from_coeffs(coeffs, l_max):
    """C_l = mean of squared Y_lm coefficients within degree l's block."""
    c_l = np.zeros(l_max + 1)
    for l in range(l_max + 1):
        start = l * l
        c_l[l] = np.mean(coeffs[start:start + 2 * l + 1] ** 2)
    return c_l


sky = Sky(nside=NSIDE)
freqs_mhz = np.linspace(FREQ_MIN, FREQ_MAX, N_POINTS)
l_vals = np.arange(1, L_MAX + 1)

D_l_grid = np.empty((N_POINTS, L_MAX))
for i, freq in enumerate(freqs_mhz):
    coeffs = sky_coefficients(sky.generate(freq), L_MAX, NSIDE)
    c_l = c_l_from_coeffs(coeffs, L_MAX)
    D_l_grid[i] = l_vals * (l_vals + 1) * c_l[1:] / (2 * np.pi)

fig, ax = plt.subplots(figsize=(10, 7))
im = ax.pcolormesh(l_vals, freqs_mhz, D_l_grid, norm=LogNorm(), shading="auto", cmap="viridis")
ax.set_xlabel("Multipole l")
ax.set_ylabel("Frequency (MHz)")
ax.set_title("Spectral-index sky: angular power vs frequency and l")
fig.colorbar(im, ax=ax, label=r"$l(l+1)C_l/2\pi$ (K$^2$)")
fig.tight_layout()

save_path = OUTPUT_DIR / "sky_model_sh.png"
fig.savefig(save_path)
print(f"Saved to {save_path}")

# Healpy cross-check at 70 MHz
ref_map = hp.ud_grade(sky.generate(70.0), NSIDE)
c_l_ours = c_l_from_coeffs(sky_coefficients(ref_map, L_MAX, NSIDE), L_MAX)
c_l_healpy = hp.anafast(ref_map, lmax=L_MAX)
print("Healpy cross-check (spectral-index sky, 70 MHz, l=1..10):")
for l in range(1, 11):
    rel_diff = abs(c_l_ours[l] - c_l_healpy[l]) / c_l_healpy[l]
    print(f"  l={l:2d}  ours={c_l_ours[l]:.6g}  healpy={c_l_healpy[l]:.6g}  rel_diff={rel_diff:.4g}")
