"""Plot the 4 lowest-order (l,m) real EField patterns -- each the combined
TM+TE field, not the two mode pieces separately -- and two highly mixed
VSH linear combinations (10+ terms each) as antenna beam patterns
(|E|^2 = E_theta^2 + E_phi^2, the squared magnitude of the vector EField)
as 3D surfaces, colored by value with plasma.
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
HEIGHT = 0.75  # m (quarter-wave at 100 MHz, off the h=lambda/2 resonance)

N_THETA, N_PHI = 150, 300
# Avoid the exact poles (theta=0, pi): the (theta_hat, phi_hat) VSH
# components have a coordinate singularity there (phi itself is undefined
# at a pole), so nudge the grid slightly inward.
theta = np.linspace(1e-4, np.pi - 1e-4, N_THETA)
phi = np.linspace(0, 2 * np.pi, N_PHI)
theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

# 4 lowest-order (l,m) fields: each is the combined TM+TE field (equal
# weight) for that (l,m) -- l=1 has no vector monopole, so l=1's 3 m
# values plus the first l=2 term give 4.
lowest_order = [
    (r"$l=1,m=-1$", VshBasis([(1, -1, "TM"), (1, -1, "TE")], [1.0, 1.0])),
    (r"$l=1,m=0$", VshBasis([(1, 0, "TM"), (1, 0, "TE")], [1.0, 1.0])),
    (r"$l=1,m=1$", VshBasis([(1, 1, "TM"), (1, 1, "TE")], [1.0, 1.0])),
    (r"$l=2,m=0$", VshBasis([(2, 0, "TM"), (2, 0, "TE")], [1.0, 1.0])),
]

# Two highly mixed linear combinations (12 terms each, l up to 4, TM+TE)
all_lm_mode = [(l, m, mode) for l in range(1, 5) for m in range(-l, l + 1) for mode in ("TM", "TE")]


def random_mix(seed: int, n_terms: int = 12) -> VshBasis:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_lm_mode), size=n_terms, replace=False)
    terms = [all_lm_mode[i] for i in idx]
    coeffs = rng.uniform(-1.0, 1.0, size=n_terms)
    return VshBasis(terms, coeffs)


mixed = [
    ("Mixed A (12 terms)", random_mix(1)),
    ("Mixed B (12 terms)", random_mix(2)),
]

panels = lowest_order + mixed  # 6 panels, filled row-major into a 2x3 grid
efields = [(title, EField.from_basis(basis, FREQUENCY, HEIGHT)) for title, basis in panels]

save_dir = OUTPUT_DIR / "tests"
save_dir.mkdir(parents=True, exist_ok=True)


def plot_panels(value_fn, save_name: str, colorbar_label: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={"projection": "3d"})

    for ax, (title, efield) in zip(axes.flat, efields):
        e_theta, e_phi = value_fn(efield)
        r = e_theta**2 + e_phi**2  # |E|^2, already non-negative

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
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.05, label=colorbar_label)

    fig.tight_layout()

    save_path = save_dir / save_name
    fig.savefig(save_path)
    print(f"Saved to {save_path}")


# Ground-reflected beam pattern: |E|^2 with image theory applied
plot_panels(
    lambda efield: efield.apply_image_theory(theta_grid, phi_grid),
    "beam_visual.png",
    "Beam pattern |E|^2",
)

# Raw beam pattern: |E|^2 unconstrained by the image-theory ground reflection
plot_panels(
    lambda efield: efield.evaluate(theta_grid, phi_grid),
    "beam_visual_raw.png",
    "Beam pattern |E|^2",
)
