"""Scatter-plot visualisation of random injection-recovery results.

Loads outputs/random_recovery_log_poly.npz and outputs/random_recovery_sync_ff.npz
produced by doRecovery.py (MAP estimation mode).

Output PNGs (all in outputs/tests/):
  random_recovery_amplitude.png  — recovered vs injected A_mk (log-log scatter)
  random_recovery_frequency.png  — recovered vs injected nu0 scatter
  random_recovery_width.png      — recovered vs injected sigma scatter
  random_recovery_bias.png       — bias histograms for all three parameters
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from forward_model.tools.config import PROJECT_ROOT
from forward_model.inference.tools.config import OUTPUT_DIR

plt.style.use(PROJECT_ROOT / 'data' / 'sofa.mplstyle')

MODELS       = ['log_poly', 'sync_ff']
MODEL_LABELS = {'log_poly': 'log-poly', 'sync_ff': 'sync+ff'}
COLORS       = {'log_poly': 'C0', 'sync_ff': 'C1'}

save_dir = OUTPUT_DIR / 'tests'
save_dir.mkdir(parents=True, exist_ok=True)

data = {}
for m in MODELS:
    data[m] = np.load(OUTPUT_DIR / f'random_recovery_{m}.npz')

# Shared colour normalisation for nu0 and A_mk colouring
nu0_all   = data[MODELS[0]]['nu0_true']
A_mk_all  = np.where(data[MODELS[0]]['A_mk_true'] > 0,
                      data[MODELS[0]]['A_mk_true'], np.nan)

nu0_norm   = mcolors.Normalize(vmin=nu0_all.min(), vmax=nu0_all.max())
Amk_norm   = mcolors.LogNorm(vmin=10, vmax=500)

# ==========================================================================
# 1. Amplitude recovery
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Signal amplitude recovery (MAP, 10–150 MHz)', fontsize=12)

for col, m in enumerate(MODELS):
    ax  = axes[col]
    d   = data[m]
    nz  = ~d['is_zero']   # non-zero trials

    sc = ax.scatter(d['A_mk_true'][nz], d['A_mk_detected'][nz],
                    c=d['nu0_true'][nz], cmap='plasma', norm=nu0_norm,
                    s=12, alpha=0.6, label='non-zero trials')

    # A=0 trials: show detected value (should be 0 after threshold)
    n_zero = d['is_zero'].sum()
    if n_zero > 0:
        ax.scatter([5]*n_zero, d['A_mk_detected'][d['is_zero']],
                   marker='^', c='k', s=14, alpha=0.5, label=f'A_inj=0 ({n_zero})')

    lim = [8, 600]
    ax.plot(lim, lim, 'k--', lw=0.8, label='1:1')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'Injected $A$ [mK]')
    ax.set_ylabel(r'Recovered $A$ [mK]')
    ax.set_title(MODEL_LABELS[m])
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label=r'$\nu_0$ [MHz]', pad=0.02)

    bias_detected = d['A_mk_detected'][nz] - d['A_mk_true'][nz]
    med_bias = float(np.median(np.abs(bias_detected)))
    det_rate = float((d['detection_snr'][nz] >= float(d['detection_threshold'])).mean()) * 100
    ax.annotate(f'median |bias| = {med_bias:.1f} mK\ndetection rate = {det_rate:.0f}%',
                xy=(0.05, 0.95), xycoords='axes fraction',
                va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

fig.tight_layout()
fig.savefig(save_dir / 'random_recovery_amplitude.png', bbox_inches='tight')
print('Saved random_recovery_amplitude.png')
plt.close(fig)


# ==========================================================================
# 2. Frequency recovery
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Signal frequency recovery (MAP, 10–150 MHz)', fontsize=12)

for col, m in enumerate(MODELS):
    ax = axes[col]
    d  = data[m]
    nz = ~d['is_zero']

    sc = ax.scatter(d['nu0_true'][nz], d['nu0_map'][nz],
                    c=np.where(d['A_mk_true'][nz]>0, d['A_mk_true'][nz], np.nan),
                    cmap='viridis', norm=Amk_norm, s=12, alpha=0.6)

    lo, hi = 10, 150
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, label='1:1')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r'Injected $\nu_0$ [MHz]')
    ax.set_ylabel(r'Recovered $\nu_0$ [MHz]')
    ax.set_title(MODEL_LABELS[m])
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label=r'$A$ [mK]', pad=0.02)

    med_bias = float(np.median(np.abs(d['bias_nu0'][nz])))
    ax.annotate(f'median |bias| = {med_bias:.1f} MHz',
                xy=(0.05, 0.95), xycoords='axes fraction',
                va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

fig.tight_layout()
fig.savefig(save_dir / 'random_recovery_frequency.png', bbox_inches='tight')
print('Saved random_recovery_frequency.png')
plt.close(fig)


# ==========================================================================
# 3. Width recovery
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Signal width recovery (MAP, 10–150 MHz)', fontsize=12)

for col, m in enumerate(MODELS):
    ax = axes[col]
    d  = data[m]
    nz = ~d['is_zero']

    sc = ax.scatter(d['sigma_true'][nz], d['sigma_map'][nz],
                    c=np.where(d['A_mk_true'][nz]>0, d['A_mk_true'][nz], np.nan),
                    cmap='viridis', norm=Amk_norm, s=12, alpha=0.6)

    lo, hi = 0, 35
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, label='1:1')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r'Injected $\sigma$ [MHz]')
    ax.set_ylabel(r'Recovered $\sigma$ [MHz]')
    ax.set_title(MODEL_LABELS[m])
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label=r'$A$ [mK]', pad=0.02)

    med_bias = float(np.median(np.abs(d['bias_sigma'][nz])))
    ax.annotate(f'median |bias| = {med_bias:.1f} MHz',
                xy=(0.05, 0.95), xycoords='axes fraction',
                va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

fig.tight_layout()
fig.savefig(save_dir / 'random_recovery_width.png', bbox_inches='tight')
print('Saved random_recovery_width.png')
plt.close(fig)


# ==========================================================================
# 4. Bias distributions (histograms)
# ==========================================================================
fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex='row')
fig.suptitle('Bias distributions (non-zero-amplitude trials, MAP, 10–150 MHz)', fontsize=12)

bias_defs = [
    ('bias_A',     r'Bias $A$ [mK]',        'mK'),
    ('bias_nu0',   r'Bias $\nu_0$ [MHz]',  'MHz'),
    ('bias_sigma', r'Bias $\sigma$ [MHz]', 'MHz'),
]

for row, (key, xlabel, unit) in enumerate(bias_defs):
    for col, m in enumerate(MODELS):
        ax = axes[row, col]
        d  = data[m]
        nz = ~d['is_zero']
        vals = d[key][nz]

        ax.hist(vals, bins=40, color=COLORS[m], alpha=0.8)
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.axvline(float(np.median(vals)), color='r', lw=1.2, ls='-', label=f'median={np.median(vals):.2g}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Count')
        ax.set_title(f'{MODEL_LABELS[m]}')
        ax.legend(fontsize=8)

        q25, q75 = np.percentile(vals, [25, 75])
        ax.annotate(f'IQR = [{q25:.2g}, {q75:.2g}] {unit}',
                    xy=(0.97, 0.95), xycoords='axes fraction',
                    va='top', ha='right', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

fig.tight_layout()
fig.savefig(save_dir / 'random_recovery_bias.png', bbox_inches='tight')
print('Saved random_recovery_bias.png')
plt.close(fig)
