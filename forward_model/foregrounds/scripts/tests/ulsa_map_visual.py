"""Plot the ULSA-generated sky map (data/exp25MHz_sky_map_with_absorption.hdf5)
via mollview, plasma colormap, percentile-clipped color scale.
"""
import h5py
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from forward_model.tools.config import PROJECT_ROOT
from forward_model.foregrounds.tools.config import OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

hdf5_path = PROJECT_ROOT / "data" / "exp25MHz_sky_map_with_absorption.hdf5"
with h5py.File(hdf5_path, "r") as f:
    sky_map = f["data"][:]

vmin, vmax = np.percentile(sky_map, [1, 99])

fig = plt.figure(figsize=(8, 6))
hp.mollview(sky_map, fig=fig, cmap="plasma", min=vmin, max=vmax,
            title="ULSA sky map with absorption, f=25 MHz", unit="K")

save_dir = OUTPUT_DIR / "tests"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "ulsa_map_visual.png"
fig.savefig(save_path)
print(f"Saved to {save_path}")
