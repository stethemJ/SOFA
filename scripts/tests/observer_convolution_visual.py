"""Verify tools/Observer.py + tools/Analyze.py: rotate the GSM2008 sky
into a real observer's local horizontal frame, mask below-horizon
pixels to ground temperature, and convolve with an example beam over
40-130 MHz. Includes a horizon-fraction sanity check, a mollview
comparison (raw vs. local/masked sky), and a brute-force pixel-space
cross-check against the fast Parseval method.
"""
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from tools.Sky import Sky
from tools.VshBasis import VshBasis
from tools.EField import EField
from tools.Observer import Observer
from tools.Analyze import Analyze
from tools.Convolution import beam_power_coefficients, antenna_temperature, sky_coefficients
from tools.config import PROJECT_ROOT, OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

FREQ_MIN, FREQ_MAX = 40.0, 130.0
N_POINTS = 50
HEIGHT = 1.0  # m
L_MAX = 40
NSIDE = 128
REFERENCE_FREQ_MHZ = 100.0

observer = Observer(latitude=-26.7, longitude=116.7, elevation=300, time="2024-06-21 12:00:00")

# Sanity check: a flat horizon always divides the sky exactly in half
theta_gal, phi_gal, above_horizon = observer._get_galactic_mapping(NSIDE)
horizon_fraction = above_horizon.mean()
print(f"Above-horizon fraction: {horizon_fraction:.4f} (expect ~0.5)")

# Visual: raw vs. local/masked sky at 100 MHz
sky = Sky(model="2008")
raw_map = sky.generate(REFERENCE_FREQ_MHZ)
local_map = observer.local_sky(raw_map, NSIDE)

fig1 = plt.figure(figsize=(8, 8))
hp.mollview(hp.ud_grade(raw_map, NSIDE), sub=(2, 1, 1), fig=fig1,
            title="Raw GSM2008 (Galactic frame), f=100 MHz", unit="K")
hp.mollview(local_map, sub=(2, 1, 2), fig=fig1,
            title="Local sky (observer frame, horizon-masked to 300 K)", unit="K")

save_dir = OUTPUT_DIR / "tests"
save_dir.mkdir(parents=True, exist_ok=True)
sky_comparison_path = save_dir / "observer_sky_comparison.png"
fig1.savefig(sky_comparison_path)
print(f"Saved to {sky_comparison_path}")

# Spectrum: single TM dipole beam, l=1 m=1
basis = VshBasis([(1, 1, "TM")], [1.0])
analyze = Analyze(basis, HEIGHT, observer, sky_model="2008", l_max=L_MAX, nside=NSIDE)

freqs_mhz = np.linspace(FREQ_MIN, FREQ_MAX, N_POINTS)
T_values = analyze.spectrum(freqs_mhz)

# Brute-force pixel-space cross-check at the reference frequency
ef_ref = EField.from_basis(basis, REFERENCE_FREQ_MHZ * 1e6, HEIGHT)
npix = hp.nside2npix(NSIDE)
theta, phi = hp.pix2ang(NSIDE, np.arange(npix))
pixel_area = 4 * np.pi / npix
e_theta, e_phi = ef_ref.apply_image_theory(theta, phi)
beam_power = e_theta**2 + e_phi**2
T_brute = np.sum(beam_power * local_map * pixel_area) / np.sum(beam_power * pixel_area)

beam_coeffs_ref = beam_power_coefficients(ef_ref, l_max=L_MAX, nside=NSIDE)
sky_coeffs_ref = sky_coefficients(local_map, l_max=L_MAX, nside=NSIDE)
T_fast = antenna_temperature(beam_coeffs_ref, sky_coeffs_ref)
rel_diff = abs(T_fast - T_brute) / abs(T_brute)

print(f"\nBrute-force cross-check (f={REFERENCE_FREQ_MHZ:.0f} MHz, local/masked sky):")
print(f"  fast (Parseval):  T_antenna = {T_fast:.4f} K")
print(f"  brute force:      T_antenna = {T_brute:.4f} K")
print(f"  relative diff:    {rel_diff:.4g}")

# Plot the spectrum
fig2, ax = plt.subplots(figsize=(10, 6))
ax.plot(freqs_mhz, T_values)
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Antenna Temperature (K)")
ax.set_title(f"Realistic antenna temperature vs frequency (MRO, GSM2008, h={HEIGHT} m)")
ax.grid(True)
ax.set_xlim(FREQ_MIN, FREQ_MAX)
fig2.tight_layout()

spectrum_path = save_dir / "observer_convolution_visual.png"
fig2.savefig(spectrum_path)
print(f"Saved to {spectrum_path}")
