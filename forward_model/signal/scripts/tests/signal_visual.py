"""Plot the Gaussian signal spectrum for default and varied parameters."""
import numpy as np
import matplotlib.pyplot as plt

from forward_model.signal.tools.Signal import GaussianSignal
from forward_model.tools.config import PROJECT_ROOT
from forward_model.signal.tools.config import OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

FREQ_MIN, FREQ_MAX = 10.0, 200.0
nu = np.linspace(FREQ_MIN, FREQ_MAX, 1000)

signals = [
    GaussianSignal(A_mk=200.0, nu0=78.0,  sigma=10.0),
    GaussianSignal(A_mk=500.0, nu0=100.0, sigma=20.0),
    GaussianSignal(A_mk=100.0, nu0=50.0,  sigma=5.0),
]

fig, ax = plt.subplots(figsize=(10, 5))
for sig in signals:
    label = rf"$A$={sig.A_mk:.0f} mK, $\nu_0$={sig.nu0:.0f} MHz, $\sigma$={sig.sigma:.0f} MHz"
    ax.plot(nu, sig.temperature(nu) * 1e3, label=label)  # convert K -> mK for display

ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Signal Temperature (mK)")
ax.set_title("Injected Gaussian Signal Profiles")
ax.legend()
ax.grid(True)

save_dir = OUTPUT_DIR / "tests"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "signal_visual.png"
fig.savefig(save_path, bbox_inches="tight")
print(f"Saved to {save_path}")
