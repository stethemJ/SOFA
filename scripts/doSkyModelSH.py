"""Compute and compare the angular power spectrum (C_l) of the sky as
seen by GSM2008, GSM2016, and LFSM (tools/Sky.py), projecting each
healpix map onto real spherical harmonics via tools/ShBasis.py's
real_sph_harm. Plots each model individually, then all three overlaid.
"""
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from tools.Sky import Sky
from tools.ShBasis import real_sph_harm
from tools.config import PROJECT_ROOT, OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

FREQUENCY = 100.0  # MHz
L_MAX = 20

MODEL_LABELS = {"2008": "GSM2008", "2016": "GSM2016", "lfsm": "LFSM"}
MODEL_KWARGS = {"2008": {}, "2016": {"resolution": "low"}, "lfsm": {}}


def angular_power_spectrum(map_data: np.ndarray, l_max: int) -> np.ndarray:
    """C_l via direct real-spherical-harmonic projection, using equal-area
    healpix pixel quadrature (accurate for l_max << nside)."""
    nside = hp.npix2nside(len(map_data))
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    pixel_area = 4 * np.pi / npix

    c_l = np.zeros(l_max + 1)
    for l in range(l_max + 1):
        a_lm_sq_sum = 0.0
        for m in range(-l, l + 1):
            y_lm = real_sph_harm(l, m, theta, phi)
            a_lm = np.sum(map_data * y_lm) * pixel_area
            a_lm_sq_sum += a_lm**2
        c_l[l] = a_lm_sq_sum / (2 * l + 1)
    return c_l


def plot_power_spectrum(model: str, c_l: np.ndarray) -> None:
    label = MODEL_LABELS[model]
    l = np.arange(1, L_MAX + 1)
    d_l = l * (l + 1) * c_l[1:] / (2 * np.pi)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(l, d_l, marker="o", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("Multipole l")
    ax.set_ylabel(r"$l(l+1)C_l / 2\pi$ (K$^2$)")
    ax.set_title(f"{label} angular power spectrum, f={FREQUENCY:.0f} MHz (C_0={c_l[0]:.4g} K^2)")
    ax.grid(True)
    fig.tight_layout()

    save_path = OUTPUT_DIR / f"sky_model_sh_{model}.png"
    fig.savefig(save_path)
    print(f"Saved to {save_path}")


results = {}
for model, kwargs in MODEL_KWARGS.items():
    sky = Sky(model=model, **kwargs)
    map_data = sky.generate(FREQUENCY)
    c_l = angular_power_spectrum(map_data, L_MAX)
    results[model] = c_l
    print(f"{MODEL_LABELS[model]}: C_0 = {c_l[0]:.4g} K^2")
    plot_power_spectrum(model, c_l)

# Healpy cross-check on the cheapest model (GSM2016, low resolution)
sky_2016 = Sky(model="2016", resolution="low")
map_2016 = sky_2016.generate(FREQUENCY)
c_l_healpy = hp.anafast(map_2016, lmax=L_MAX)

print("\nHealpy cross-check (GSM2016, low resolution):")
print(f"{'l':>3}  {'C_l (ShBasis)':>16}  {'C_l (healpy)':>16}  {'rel. diff':>10}")
for l in range(1, 11):
    ours = results["2016"][l]
    theirs = c_l_healpy[l]
    rel_diff = abs(ours - theirs) / theirs
    print(f"{l:>3}  {ours:>16.6g}  {theirs:>16.6g}  {rel_diff:>10.4g}")

# Combined overlay
fig, ax = plt.subplots(figsize=(10, 7))
l = np.arange(1, L_MAX + 1)
for model in MODEL_KWARGS:
    c_l = results[model]
    d_l = l * (l + 1) * c_l[1:] / (2 * np.pi)
    ax.plot(l, d_l, marker="o", markersize=3, label=MODEL_LABELS[model])

ax.set_yscale("log")
ax.set_xlabel("Multipole l")
ax.set_ylabel(r"$l(l+1)C_l / 2\pi$ (K$^2$)")
ax.set_title(f"Angular power spectrum comparison, f={FREQUENCY:.0f} MHz")
ax.legend()
ax.grid(True)
fig.tight_layout()

save_path = OUTPUT_DIR / "sky_model_sh_comparison.png"
fig.savefig(save_path)
print(f"Saved to {save_path}")
