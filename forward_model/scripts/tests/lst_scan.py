"""24-hour LST scan: animate the visible sky rotating overhead across
24 evenly-spaced sidereal-time blocks, and compute the LST-averaged
antenna temperature spectrum (l=1, m=-1 TM dipole) over 40-130 MHz,
fit with FitFuncs.log_poly_exp.
"""
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from PIL import Image

from forward_model.foregrounds.tools.Sky import Sky
from antenna.tools.VshBasis import VshBasis
from forward_model.tools.Observer import Observer
from forward_model.tools.Analyze import Analyze
from forward_model.foregrounds.tools.FitFuncs import FitFuncs
from forward_model.tools.config import PROJECT_ROOT
from antenna.tools.config import OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

LATITUDE, LONGITUDE, ELEVATION = 68.6, -149.6, 720  # Toolik Field Station, Alaska
START_TIME = "2026-08-26 12:00:00"
N_BLOCKS = 24
HEIGHT = 1.0  # m
L_MAX = 40
NSIDE = 128
GIF_FREQ_MHZ = 30.0
FREQ_MIN, FREQ_MAX = 40.0, 130.0
N_POINTS = 50
FIT_N = 5

save_dir = OUTPUT_DIR / "tests"
save_dir.mkdir(parents=True, exist_ok=True)

observers = Observer.lst_scan(LATITUDE, LONGITUDE, ELEVATION, START_TIME, n_blocks=N_BLOCKS)

# --- GIF: visible sky rotating overhead ---
sky = Sky(nside=NSIDE)
raw_map = sky.generate(GIF_FREQ_MHZ)
local_maps = [obs.local_sky(raw_map, NSIDE) for obs in observers]

_, _, above_horizon = observers[0]._get_galactic_mapping(NSIDE)
above_values = np.concatenate([m[above_horizon] for m in local_maps])
# Clip color scale to 1st-99th percentile: a small fraction of pixels
# can otherwise dominate the range and wash out real sky structure.
vmin, vmax = np.percentile(above_values, [1, 99])

frame_paths = []
for i, local_map in enumerate(local_maps):
    hp.orthview(local_map, rot=(0, 90, 0), half_sky=True, min=vmin, max=vmax,
                title=f"LST block {i}/{N_BLOCKS}", unit="K", cmap="plasma")
    frame_path = save_dir / f"_lst_frame_{i:02d}.png"
    plt.savefig(frame_path)
    plt.close("all")
    frame_paths.append(frame_path)

frames = [Image.open(p) for p in frame_paths]
gif_path = save_dir / "lst_scan.gif"
frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=200, loop=0)
for p in frame_paths:
    p.unlink()
print(f"Saved to {gif_path}")

# --- LST-averaged spectrum + fit ---
basis = VshBasis([(1, -1, "TM")], [1.0])
analyze = Analyze(basis, HEIGHT, observers[0], l_max=L_MAX, nside=NSIDE)

freqs_mhz = np.linspace(FREQ_MIN, FREQ_MAX, N_POINTS)
nu_ref = np.sqrt(FREQ_MIN * FREQ_MAX)

T_avg = analyze.lst_averaged_spectrum(freqs_mhz, observers)
T_ref = analyze.lst_averaged_spectrum([nu_ref], observers)[0]

# Sanity check: confirm the 24 LST blocks genuinely differ
idx_mid = len(freqs_mhz) // 2
T_stack_mid = [Analyze(basis, HEIGHT, obs, l_max=L_MAX, nside=NSIDE)
               .antenna_temperature(freqs_mhz[idx_mid]) for obs in observers]
print(f"Std dev across 24 LST blocks at {freqs_mhz[idx_mid]:.1f} MHz: {np.std(T_stack_mid):.4g} K "
      f"(mean {np.mean(T_stack_mid):.4g} K)")

ff = FitFuncs(nu_ref, T_ref)
coeffs = ff.fit_log_poly_exp(freqs_mhz, T_avg, FIT_N)
T_fit = ff.log_poly_exp(freqs_mhz, coeffs)
resid = T_avg - T_fit
rms = np.sqrt(np.mean(resid**2))
print(f"RMS residual: {rms:.4g} K")

fig, (ax_fit, ax_res) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, height_ratios=[3, 1])
ax_fit.plot(freqs_mhz, T_avg, label="LST-averaged T_antenna", color="black", lw=1)
ax_fit.plot(freqs_mhz, T_fit, label="log_poly_exp fit", linestyle="--")
ax_fit.set_ylabel("Antenna Temperature (K)")
ax_fit.set_title(f"LST-averaged antenna temperature (l=1,m=-1 TM dipole, N={FIT_N}, RMS={rms:.3g} K)")
ax_fit.legend()
ax_fit.grid(True)

ax_res.plot(freqs_mhz, resid, color="firebrick")
ax_res.axhline(0, color="gray", lw=0.8)
ax_res.set_xlabel("Frequency (MHz)")
ax_res.set_ylabel("Residual (K)")
ax_res.grid(True)
ax_res.set_xlim(FREQ_MIN, FREQ_MAX)

fig.tight_layout()
spectrum_path = save_dir / "lst_scan_spectrum.png"
fig.savefig(spectrum_path)
print(f"Saved to {spectrum_path}")
