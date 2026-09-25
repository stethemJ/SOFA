"""Plot the spectral-index foreground sky map at 30 MHz (plasma colormap)
and the per-pixel spectral index β map."""
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from forward_model.foregrounds.tools.Sky import Sky
from forward_model.tools.config import PROJECT_ROOT
from forward_model.foregrounds.tools.config import OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

FREQUENCY_MHZ = 30.0
HSPACE = 0.5  # vertical gap between panels — increase if titles/colorbars overlap

sky = Sky()
sky_map = sky.generate(FREQUENCY_MHZ)
beta_map = sky._beta

fig = plt.figure(figsize=(10, 12))

vmin, vmax = np.percentile(sky_map, [1, 99])
hp.mollview(sky_map, sub=(2, 1, 1), fig=fig, cmap="plasma",
            min=vmin, max=vmax,
            title=f"Spectral Index Built Sky, f={FREQUENCY_MHZ:.0f} MHz", unit="K")

bmin, bmax = np.percentile(beta_map, [1, 99])
hp.mollview(beta_map, sub=(2, 1, 2), fig=fig, cmap="RdYlBu_r",
            min=bmin, max=bmax,
            title=r"Per-pixel spectral index $\beta$ (GSM2016, 230/408 MHz)")

fig.subplots_adjust(hspace=HSPACE)

save_dir = OUTPUT_DIR / "tests"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "sky_model_visual.png"
fig.savefig(save_path, bbox_inches="tight")
print(f"Saved to {save_path}")
