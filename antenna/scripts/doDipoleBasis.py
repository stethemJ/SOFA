"""Plot a realistic horizontal dipole antenna's beam pattern (l=1, m=1,
TM only -- a real linear dipole is purely electric-current, so it uses
the electric multipole (TM) VSH term, not the magnetic (TE) one). Two
panels: free-space (no image theory) vs. ground-reflected (image theory
applied), showing the horizon null + zenith lobe.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from antenna.tools.VshBasis import VshBasis
from antenna.tools.EField import EField
from forward_model.tools.config import PROJECT_ROOT
from antenna.tools.config import OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

FREQUENCY = 100e6  # Hz
HEIGHT = 0.75  # m (quarter-wave at 100 MHz)

N_THETA, N_PHI = 150, 300
theta = np.linspace(1e-4, np.pi - 1e-4, N_THETA)
phi = np.linspace(0, 2 * np.pi, N_PHI)
theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

basis = VshBasis([(1, 1, "TM")], [1.0])
efield = EField.from_basis(basis, FREQUENCY, HEIGHT)

panels = [
    ("Free space (no image theory)", efield.evaluate(theta_grid, phi_grid)),
    ("Ground-reflected (image theory)", efield.apply_image_theory(theta_grid, phi_grid)),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={"projection": "3d"})

for ax, (title, (e_theta, e_phi)) in zip(axes, panels):
    r = e_theta**2 + e_phi**2  # |E|^2, beam pattern

    x = r * np.sin(theta_grid) * np.cos(phi_grid)
    y = r * np.sin(theta_grid) * np.sin(phi_grid)
    z = r * np.cos(theta_grid)

    norm = colors.Normalize(vmin=r.min(), vmax=r.max())
    facecolors = cm.plasma(norm(r))

    ax.plot_surface(
        x, y, z,
        facecolors=facecolors,
        rstride=1, cstride=1,
        linewidth=0, antialiased=True, shade=False,
    )
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)

    mappable = cm.ScalarMappable(norm=norm, cmap="plasma")
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.05, label="Beam pattern |E|^2")

fig.suptitle(f"Horizontal dipole (l=1, m=1, TM) beam pattern, f={FREQUENCY/1e6:.0f} MHz, h={HEIGHT} m")
fig.tight_layout()

save_path = OUTPUT_DIR / "dipole_basis.png"
fig.savefig(save_path)
print(f"Saved to {save_path}")
