"""Run the fiducial ARES global-signal simulation, export it to
data/ares_21cm_signal.csv (a new file -- the existing
data/injected_21cm_signal.csv is left untouched), and plot it against
that existing injected signal and the EDGES flattened-Gaussian best-fit
(Bowman et al. 2018) for comparison.

Run with the dedicated ARES-compatible interpreter, from the project
root (so tools/config.py and hi21cm/tools/FitFuncs.py are importable):

    PYTHONPATH=. simulations/.venv311/bin/python3 simulations/scripts/generate_signal.py
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

from forward_model.simulations.tools.GlobalSignal import GlobalSignal, FIDUCIAL_PARAMS
from forward_model.hi21cm.tools.FitFuncs import FitFuncs
from forward_model.tools.config import PROJECT_ROOT
from forward_model.simulations.tools.config import OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

print(f"Running ARES with fiducial parameters: {FIDUCIAL_PARAMS}")
signal = GlobalSignal().run()

freq_mhz, z, trough_mk = signal.trough()
print(f"Trough: freq={freq_mhz:.2f} MHz (z={z:.2f}), depth={trough_mk:.2f} mK "
      f"(EDGES low-band target: 78 MHz, -500 mK -- standard astrophysics alone "
      f"cannot reach that depth; this fiducial model matches the location honestly, "
      f"not the amplitude)")

csv_path = PROJECT_ROOT / "data" / "ares_21cm_signal.csv"
signal.export_csv(csv_path)
print(f"Saved {csv_path}")

records = signal.to_records()
sim_freq = np.array([r["frequency_MHz"] for r in records])
sim_temp_mk = np.array([r["brightness_temp_mK"] for r in records])
mask = (sim_freq >= 10.0) & (sim_freq <= 150.0)
sim_freq, sim_temp_mk = sim_freq[mask], sim_temp_mk[mask]

with open(PROJECT_ROOT / "data" / "injected_21cm_signal.csv", newline="") as f:
    rows = [(float(r["frequency_MHz"]), float(r["brightness_temp_mK"])) for r in csv.DictReader(f)]
existing_freq, existing_temp_mk = map(np.array, zip(*rows))
order = np.argsort(existing_freq)
existing_freq, existing_temp_mk = existing_freq[order], existing_temp_mk[order]
existing_mask = (existing_freq >= 10.0) & (existing_freq <= 150.0)

ff = FitFuncs(nu_ref=78.0, T_ref=1.0)
edges_freq = np.linspace(10.0, 150.0, 500)
edges_temp_mk = ff.flattened_gaussian_21cm(edges_freq, A=-500.0, nu0=78.0, w=19.0, tau=7.0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sim_freq, sim_temp_mk, label="ARES simulation (this work)", lw=1.5)
ax.plot(existing_freq[existing_mask], existing_temp_mk[existing_mask],
        label="Existing injected_21cm_signal.csv", linestyle="--")
ax.plot(edges_freq, edges_temp_mk, label="EDGES flattened-Gaussian fit\n(Bowman et al. 2018)", linestyle=":")
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Brightness Temperature (mK)")
ax.set_title("Simulated vs. existing 21-cm global signal")
ax.legend()
ax.grid(True)
fig.tight_layout()

save_dir = OUTPUT_DIR / "tests"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "ares_21cm_signal.png"
fig.savefig(save_path)
print(f"Saved {save_path}")
