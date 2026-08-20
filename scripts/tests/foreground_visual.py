"""Plot the GSM2008 diffuse Galactic foreground's mean sky temperature vs.
frequency, over 10-150 MHz -- the same frequency domain as 21cm_visual.py.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

from tools.Sky import Sky
from tools.config import PROJECT_ROOT, OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

csv_path = PROJECT_ROOT / "data" / "injected_21cm_signal.csv"
with open(csv_path, newline="") as f:
    freq_mhz = np.array([float(r["frequency_MHz"]) for r in csv.DictReader(f)])

freq_mhz = np.sort(freq_mhz[(freq_mhz >= 10.0) & (freq_mhz <= 150.0)])

sky = Sky()
maps = sky.generate(freq_mhz)
mean_temp_k = maps.mean(axis=1)

save_dir = OUTPUT_DIR / "tests"
save_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(freq_mhz, mean_temp_k, marker="o", markersize=3)
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Mean Sky Temperature (K)")
ax.set_title("GSM2008 Diffuse Galactic Foreground")
ax.grid(True)
fig.tight_layout()

save_path = save_dir / "foreground_visual.png"
fig.savefig(save_path)
print(f"Saved to {save_path}")

fig_log, ax_log = plt.subplots(figsize=(10, 6))
ax_log.plot(freq_mhz, mean_temp_k, marker="o", markersize=3)
ax_log.set_xscale("log")
ax_log.set_yscale("log")
ax_log.set_xlabel("Frequency (MHz)")
ax_log.set_ylabel("Mean Sky Temperature (K)")
ax_log.set_title("GSM2008 Diffuse Galactic Foreground (log-log)")
ax_log.grid(True, which="both")
fig_log.tight_layout()

save_path_log = save_dir / "foreground_visual_loglog.png"
fig_log.savefig(save_path_log)
print(f"Saved to {save_path_log}")
