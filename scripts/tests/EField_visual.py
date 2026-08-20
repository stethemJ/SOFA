"""Plot the 4 lowest-order real Ylm basis functions and two highly mixed
Ylm linear combinations (10+ terms each) as EField ground-reflected 3D
surfaces (image theory), radius = |value|, colored by the signed value
with plasma.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from tools.ShBasis import ShBasis
from tools.EField import EField
from tools.config import PROJECT_ROOT, OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

FREQUENCY = 100e6  # Hz
HEIGHT = 1.5  # m
POLARIZATION = "vertical"

N_THETA, N_PHI = 150, 300
theta = np.linspace(0, np.pi, N_THETA)
phi = np.linspace(0, 2 * np.pi, N_PHI)
theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

# 4 lowest-order Ylms: l=0 (1 term) + l=1 (3 terms)
lowest_order = [
    (r"$Y_0^0$", ShBasis([(0, 0)], [1.0])),
    (r"$Y_1^{-1}$", ShBasis([(1, -1)], [1.0])),
    (r"$Y_1^0$", ShBasis([(1, 0)], [1.0])),
    (r"$Y_1^1$", ShBasis([(1, 1)], [1.0])),
]

# Two highly mixed linear combinations (12 terms each, l up to 4)
all_lm = [(l, m) for l in range(5) for m in range(-l, l + 1)]


def random_mix(seed: int, n_terms: int = 12) -> ShBasis:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_lm), size=n_terms, replace=False)
    terms = [all_lm[i] for i in idx]
    coeffs = rng.uniform(-1.0, 1.0, size=n_terms)
    return ShBasis(terms, coeffs)


mixed = [
    ("Mixed A (12 terms)", random_mix(1)),
    ("Mixed B (12 terms)", random_mix(2)),
]

panels = lowest_order + mixed  # 6 panels, filled row-major into a 2x3 grid
efields = [(title, EField.from_basis(basis, FREQUENCY, HEIGHT, POLARIZATION)) for title, basis in panels]

save_dir = OUTPUT_DIR / "tests"
save_dir.mkdir(parents=True, exist_ok=True)


def plot_panels(value_fn, save_name: str, colorbar_label: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={"projection": "3d"})

    for ax, (title, efield) in zip(axes.flat, efields):
        values = value_fn(efield)
        r = np.abs(values)

        x = r * np.sin(theta_grid) * np.cos(phi_grid)
        y = r * np.sin(theta_grid) * np.sin(phi_grid)
        z = r * np.cos(theta_grid)

        norm = colors.Normalize(vmin=values.min(), vmax=values.max())
        facecolors = cm.plasma(norm(values))

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
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.05, label=colorbar_label)

    fig.tight_layout()

    save_path = save_dir / save_name
    fig.savefig(save_path)
    print(f"Saved to {save_path}")


# Ground-reflected field (image theory applied)
plot_panels(
    lambda efield: efield.apply_image_theory(theta_grid, phi_grid),
    "EField_visual.png",
    "EField value (image theory)",
)

# Raw field, unconstrained by the image-theory ground reflection
plot_panels(
    lambda efield: efield.evaluate(theta_grid, phi_grid),
    "EField_visual_raw.png",
    "EField value (raw)",
)
