"""Plot the injected 21-cm global signal's mean brightness temperature vs.
frequency, over 10-150 MHz at 0.5 MHz spacing (the analysis grid shared
with foregrounds/scripts/tests/foreground_visual.py). The signal is
spatially uniform (a monopole -- see hi21cm/tools/Sky.py's
inject_21cm), so its sky-average equals the tabulated value itself;
read hi21cm.tools.Sky.CSV_PATH directly rather than round tripping
through Sky/pygdsm, and interpolate onto the analysis grid since the
CSV's native (redshift-derived) sampling isn't uniform.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

from forward_model.tools.config import PROJECT_ROOT
from forward_model.hi21cm.tools.config import OUTPUT_DIR
from forward_model.hi21cm.tools.Sky import CSV_PATH

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

FREQ_MIN, FREQ_MAX, FREQ_STEP = 10.0, 150.0, 0.5  # shared with foreground_visual.py

with open(CSV_PATH, newline="") as f:
    rows = [(float(r["frequency_MHz"]), float(r["brightness_temp_mK"]))
            for r in csv.DictReader(f)]

csv_freq_mhz, csv_temp_mk = map(np.array, zip(*rows))
order = np.argsort(csv_freq_mhz)
csv_freq_mhz, csv_temp_mk = csv_freq_mhz[order], csv_temp_mk[order]

freq_mhz = np.arange(FREQ_MIN, FREQ_MAX + FREQ_STEP, FREQ_STEP)
temp_mk = np.interp(freq_mhz, csv_freq_mhz, csv_temp_mk)

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
