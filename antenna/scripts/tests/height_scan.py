"""3x3 grid of ground-reflected beam patterns (|E|^2, image theory
applied): rows are antenna height (1, 2, 3 m), columns are VSH basis
mode (l=1,m=1; l=2,m=0; a 10-term mixed combination). TM (electric
multipole) only -- a real dipole/multipole antenna has no TE content.
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
HEIGHTS = [1, 2, 3]  # m

N_THETA, N_PHI = 150, 300
theta = np.linspace(1e-4, np.pi - 1e-4, N_THETA)
phi = np.linspace(0, 2 * np.pi, N_PHI)
theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

all_lm_mode = [(l, m, "TM") for l in range(1, 5) for m in range(-l, l + 1)]


def random_mix(seed: int, n_terms: int = 10) -> VshBasis:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_lm_mode), size=n_terms, replace=False)
    terms = [all_lm_mode[i] for i in idx]
    coeffs = rng.uniform(-1.0, 1.0, size=n_terms)
    return VshBasis(terms, coeffs)


columns = [
    ("(l=1,m=1)", VshBasis([(1, 1, "TM")], [1.0])),
    ("(l=2,m=0)", VshBasis([(2, 0, "TM")], [1.0])),
    ("10-mode mixed", random_mix(1)),
]

fig, axes = plt.subplots(3, 3, figsize=(15, 15), subplot_kw={"projection": "3d"})

for i, height in enumerate(HEIGHTS):
    for j, (label, basis) in enumerate(columns):
        ax = axes[i, j]
        efield = EField.from_basis(basis, FREQUENCY, height)
        e_theta, e_phi = efield.apply_image_theory(theta_grid, phi_grid)
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
        ax.set_title(f"h={height}m, {label}")
        ax.set_box_aspect([1, 1, 1])
        ax.grid(True)

        mappable = cm.ScalarMappable(norm=norm, cmap="plasma")
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.05, label="|E|^2")

fig.suptitle(f"Beam pattern vs height and VSH mode, f={FREQUENCY/1e6:.0f} MHz (image theory applied)")
fig.tight_layout()

save_dir = OUTPUT_DIR / "tests"
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / "height_scan.png"
fig.savefig(save_path)
print(f"Saved to {save_path}")
