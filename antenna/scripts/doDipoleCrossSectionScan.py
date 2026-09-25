"""Scan the l=1, m=1 VSH beam pattern's E-plane and H-plane cross sections
over antenna height above ground, at a fixed frequency, with image theory
applied.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from antenna.tools.VshBasis import VshBasis
from antenna.tools.EField import EField
from forward_model.tools.config import PROJECT_ROOT
from antenna.tools.config import OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

FREQUENCY = 100e6  # Hz
HEIGHTS = np.linspace(1, 6, 6)  # m

theta = np.linspace(1e-4, np.pi / 2, 300)
theta_deg = np.degrees(theta)

basis = VshBasis([(1, 1, "TM")], [1.0])  # pure electric dipole -- a real dipole has no TE (magnetic) content
colors_list = cm.tab10.colors  # 10 discrete, well-separated colors

fig, (ax_e, ax_h) = plt.subplots(1, 2, figsize=(16, 7))

for color, h in zip(colors_list, HEIGHTS):
    ef = EField.from_basis(basis, FREQUENCY, h)
    label = f"h={h:.2f} m"

    e_theta, e_phi = ef.apply_image_theory(theta, np.zeros_like(theta))  # phi=0, E-plane
    ax_e.plot(theta_deg, e_theta**2 + e_phi**2, color=color, label=label)

    e_theta, e_phi = ef.apply_image_theory(theta, np.full_like(theta, np.pi / 2))  # phi=90, H-plane
    ax_h.plot(theta_deg, e_theta**2 + e_phi**2, color=color, label=label)

for ax, title in ((ax_e, "E-plane (phi=0)"), (ax_h, "H-plane (phi=90)")):
    ax.set_xlabel("theta (deg, 0=zenith, 90=horizon)")
    ax.set_ylabel("|E|^2")
    ax.set_title(title)
    ax.grid(True)

ax_h.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

fig.suptitle(f"Dipole (l=1, m=1) E/H-plane cross sections vs height, f={FREQUENCY/1e6:.0f} MHz")
fig.tight_layout()

save_path = OUTPUT_DIR / "dipole_cross_section_scan.png"
fig.savefig(save_path)
print(f"Saved to {save_path}")
