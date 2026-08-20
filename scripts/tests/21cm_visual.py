"""Plot the injected 21-cm global signal's mean brightness temperature vs.
frequency, over 10-150 MHz. The signal is spatially uniform (a monopole --
see tools/Sky.py's inject_21cm), so its sky-average equals the tabulated
value itself; read data/injected_21cm_signal.csv directly rather than
round tripping through Sky/pygdsm.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

from tools.config import PROJECT_ROOT, OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

csv_path = PROJECT_ROOT / "data" / "injected_21cm_signal.csv"
with open(csv_path, newline="") as f:
    rows = [(float(r["frequency_MHz"]), float(r["brightness_temp_mK"]))
            for r in csv.DictReader(f)]

freq_mhz, temp_mk = map(np.array, zip(*rows))
mask = (freq_mhz >= 10.0) & (freq_mhz <= 150.0)  # shared with foreground_visual.py
order = np.argsort(freq_mhz[mask])
freq_mhz = freq_mhz[mask][order]
temp_mk = temp_mk[mask][order]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(freq_mhz, temp_mk, marker="o", markersize=3)
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Brightness Temperature (mK)")
ax.set_title("Injected 21-cm Global Signal")
ax.grid(True)
fig.tight_layout()

save_dir = OUTPUT_DIR / "tests"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "21cm_visual.png"
fig.savefig(save_path)
print(f"Saved to {save_path}")
