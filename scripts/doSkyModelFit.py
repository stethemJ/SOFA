"""Fit three foreground spectral models (tools/FitFuncs.py) to the sky
average (tools/Sky.py) over 40-130 MHz, and plot the fits and residuals
for comparison. All three models are given 5 free parameters for a fair
comparison. Runs for GSM2008, GSM2016, and LFSM.
"""
import numpy as np
import matplotlib.pyplot as plt

from tools.Sky import Sky
from tools.FitFuncs import FitFuncs
from tools.config import PROJECT_ROOT, OUTPUT_DIR

plt.style.use(PROJECT_ROOT / "data" / "sofa.mplstyle")

FREQ_MIN, FREQ_MAX = 40.0, 130.0
N_POINTS = 100
N_ORDER = 4  # log_poly: N+1=5 coeffs; power_poly/log_poly_exp: N=5 coeffs -> 5-parameter comparison

MODEL_LABELS = {"2008": "GSM2008", "2016": "GSM2016", "lfsm": "LFSM"}


def run(model: str, resolution: str = None) -> dict:
    label = MODEL_LABELS[model]
    nu = np.linspace(FREQ_MIN, FREQ_MAX, N_POINTS)

    sky_kwargs = {"resolution": resolution} if resolution else {}
    sky = Sky(model=model, **sky_kwargs)
    T_data = sky.generate(nu).mean(axis=1)

    nu_ref = np.sqrt(FREQ_MIN * FREQ_MAX)
    T_ref = sky.generate(nu_ref).mean()

    ff = FitFuncs(nu_ref, T_ref)

    models = [
        ("log_poly", ff.log_poly, ff.fit_log_poly, N_ORDER),
        ("power_poly", ff.power_poly, ff.fit_power_poly, N_ORDER + 1),
        ("log_poly_exp", ff.log_poly_exp, ff.fit_log_poly_exp, N_ORDER + 1),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex="col", height_ratios=[3, 1])

    results = {}
    for col, (name, model_fn, fit_fn, N) in enumerate(models):
        coeffs = fit_fn(nu, T_data, N)
        T_fit = model_fn(nu, coeffs)
        resid = T_data - T_fit
        results[name] = (nu, resid)
        rms = np.sqrt(np.mean(resid**2))
        print(f"{label}: {name}: {len(coeffs)} params, RMS residual = {rms:.4g} K")

        ax_fit = axes[0, col]
        ax_fit.plot(nu, T_data, label=f"{label} mean", color="black", lw=1)
        ax_fit.plot(nu, T_fit, label="fit", linestyle="--")
        ax_fit.set_title(f"{name} (N={N}, RMS={rms:.3g} K)")
        ax_fit.set_ylabel("Mean Sky Temperature (K)")
        ax_fit.legend()
        ax_fit.grid(True)

        ax_res = axes[1, col]
        ax_res.plot(nu, resid, color="firebrick")
        ax_res.axhline(0, color="gray", lw=0.8)
        ax_res.set_xlabel("Frequency (MHz)")
        ax_res.set_ylabel("Residual (K)")
        ax_res.grid(True)

    fig.suptitle(f"Foreground model fits to {label} sky average, {FREQ_MIN:.0f}-{FREQ_MAX:.0f} MHz (nu_ref={nu_ref:.1f} MHz)")
    fig.tight_layout()

    save_path = OUTPUT_DIR / f"sky_model_fit_{model}.png"
    fig.savefig(save_path)
    print(f"Saved to {save_path}")

    return results


def plot_residual_comparison(*labeled_results: tuple[str, dict]) -> None:
    method_names = list(labeled_results[0][1].keys())

    all_resid = np.concatenate([
        r for _, results in labeled_results for _, r in results.values()
    ])
    ylim = (all_resid.min(), all_resid.max())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, name in zip(axes, method_names):
        for label, results in labeled_results:
            nu, resid = results[name]
            ax.plot(nu, resid, label=label)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylim(ylim)
        ax.grid(True)

    axes[0].set_ylabel("Residual (K)")
    axes[0].legend()

    model_names = ", ".join(label for label, _ in labeled_results)
    fig.suptitle(f"Fit residuals: {model_names}, {FREQ_MIN:.0f}-{FREQ_MAX:.0f} MHz")
    fig.tight_layout()

    save_path = OUTPUT_DIR / "sky_model_fit_residuals_comparison.png"
    fig.savefig(save_path)
    print(f"Saved to {save_path}")


results_2008 = run("2008")
results_2016 = run("2016", resolution="low")
results_lfsm = run("lfsm")
plot_residual_comparison(
    (MODEL_LABELS["2008"], results_2008),
    (MODEL_LABELS["2016"], results_2016),
    (MODEL_LABELS["lfsm"], results_lfsm),
)
