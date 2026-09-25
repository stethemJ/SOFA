"""Random injection-recovery validation using MAP estimation (10–150 MHz).

For each random trial:
  1. Draw (A_mk, nu0, sigma) from the parameter prior
  2. Inject a Gaussian absorption signal into the pixel-averaged foreground sky
  3. Add radiometer noise (frequency-dependent)
  4. Recover signal parameters via MAP estimation (fit_map)
  5. Record bias = recovered - injected

Outputs:
  outputs/random_recovery_log_poly.npz
  outputs/random_recovery_sync_ff.npz
"""
import numpy as np

from forward_model.foregrounds.tools.Sky import Sky as ForegroundSky
from forward_model.foregrounds.tools.FitFuncs import FitFuncs as FG
from forward_model.signal.tools.Signal import GaussianSignal
from forward_model.inference.tools.Fitter import BayesianFitter
from forward_model.inference.tools.config import OUTPUT_DIR

# ── Observation band ─────────────────────────────────────────────────────────
FREQ_MIN, FREQ_MAX = 10.0, 150.0   # MHz
N_POINTS = 300
LOG_POLY_ORDER = 5   # higher order for wider band

# ── Radiometer noise ─────────────────────────────────────────────────────────
T_INT_HOURS  = 1000.0
DELTA_NU_MHZ = 1.0

# ── Random trial config ──────────────────────────────────────────────────────
N_TRIALS            = 500
FRAC_ZERO           = 0.1   # fraction of trials with A_mk = 0 (pure foreground)
RNG_SEED            = 42
DETECTION_THRESHOLD = 2.0   # SNR = A_mk_map / sigma_A_eff; below this → A_mk = 0

# ── Build foreground sky ─────────────────────────────────────────────────────
nu     = np.linspace(FREQ_MIN, FREQ_MAX, N_POINTS)
nu_ref = np.sqrt(FREQ_MIN * FREQ_MAX)   # geometric mean ≈ 38.7 MHz

print("Building foreground sky (this may take ~60 s for 10–150 MHz)...")
sky           = ForegroundSky()
sky_maps_base = sky.generate(nu)           # (N_POINTS, npix)
T_fg_base     = sky_maps_base.mean(axis=1) # (N_POINTS,) clean fg spectrum
T_ref = float(T_fg_base[np.argmin(np.abs(nu - nu_ref))])

t_int_s     = T_INT_HOURS * 3600.0
delta_nu_hz = DELTA_NU_MHZ * 1e6
sigma_noise = T_fg_base / np.sqrt(delta_nu_hz * t_int_s)   # (N_POINTS,) [K]
print(f"Noise: {sigma_noise.min()*1e3:.2f}–{sigma_noise.max()*1e3:.2f} mK")

# ── Oracle foreground reference ──────────────────────────────────────────────
fg_func   = FG(nu_ref, T_ref)
oracle_lp = fg_func.fit_log_poly(nu, T_fg_base, LOG_POLY_ORDER)
oracle_sf = fg_func.fit_sync_ff(nu, T_fg_base)
print(f"Oracle sync_ff: A_s={oracle_sf[0]:.1f} K  beta={oracle_sf[1]:.3f}  Te={oracle_sf[2]:.0f} K  EM={oracle_sf[3]:.2f}")

# ── Draw random injection parameters ─────────────────────────────────────────
rng = np.random.default_rng(RNG_SEED)
is_zero    = rng.random(N_TRIALS) < FRAC_ZERO
A_mk_true  = np.where(is_zero, 0.0,
                       10.0 ** rng.uniform(np.log10(10), np.log10(500), N_TRIALS))
nu0_true   = rng.uniform(FREQ_MIN + 15, FREQ_MAX - 15, N_TRIALS)
sigma_true = rng.uniform(2.0, 30.0, N_TRIALS)
print(f"Trials: {N_TRIALS}  (zero-amplitude: {is_zero.sum()})\n")

# ── Run recovery for each foreground model ───────────────────────────────────
FG_PARAMS_N = {'log_poly': LOG_POLY_ORDER + 1, 'sync_ff': 4}
FG_ORACLE   = {'log_poly': oracle_lp, 'sync_ff': oracle_sf}

for fg_model in ('log_poly', 'sync_ff'):
    print(f"--- {fg_model} ---")
    n_fg = FG_PARAMS_N[fg_model]

    A_mk_map      = np.zeros(N_TRIALS)
    A_mk_detected = np.zeros(N_TRIALS)   # hard-zero for SNR < DETECTION_THRESHOLD
    nu0_map       = np.zeros(N_TRIALS)
    sigma_map     = np.zeros(N_TRIALS)
    sigma_A_eff   = np.zeros(N_TRIALS)
    detection_snr = np.zeros(N_TRIALS)
    fg_map        = np.zeros((N_TRIALS, n_fg))
    res_rms       = np.zeros(N_TRIALS)
    success       = np.zeros(N_TRIALS, dtype=bool)

    for i in range(N_TRIALS):
        sig    = GaussianSignal(A_mk=A_mk_true[i], nu0=nu0_true[i], sigma=sigma_true[i])
        T_clean = sig.inject(sky_maps_base, nu).mean(axis=1)
        noise_i = rng.normal(0.0, sigma_noise)
        T_data  = T_clean + noise_i

        fitter = BayesianFitter(nu, T_data, nu_ref, T_ref, sigma_noise=sigma_noise)
        result = fitter.fit_map(fg_model, N=LOG_POLY_ORDER, T_fg_ref=T_fg_base)

        A_mk_map[i]      = result['A_mk_map']
        A_mk_detected[i] = result['A_mk_map'] if result['detection_snr'] >= DETECTION_THRESHOLD else 0.0
        nu0_map[i]       = result['nu0_map']
        sigma_map[i]     = result['sigma_map']
        sigma_A_eff[i]   = result['sigma_A_eff']
        detection_snr[i] = result['detection_snr']
        fg_map[i]        = result['fg_params_map']
        res_rms[i]       = result['residuals_rms']
        success[i]       = result['success']

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N_TRIALS}]  last bias A={A_mk_map[i]-A_mk_true[i]:+.1f} mK  "
                  f"bias nu0={nu0_map[i]-nu0_true[i]:+.1f} MHz  success={success[i]}")

    # non-zero trials for bias stats
    nz = ~is_zero
    bias_A     = A_mk_map   - A_mk_true
    bias_nu0   = nu0_map    - nu0_true
    bias_sigma = sigma_map  - sigma_true

    detected_as_zero = detection_snr < DETECTION_THRESHOLD
    print(f"  Non-zero trials ({nz.sum()}):  "
          f"median |bias A|={np.median(np.abs(bias_A[nz])):.1f} mK  "
          f"median |bias nu0|={np.median(np.abs(bias_nu0[nz])):.1f} MHz")
    print(f"  Detection rate (non-zero): {(~detected_as_zero[nz]).sum()}/{nz.sum()} "
          f"({100*(~detected_as_zero[nz]).mean():.1f}%)")
    print(f"  Zero-A trials ({is_zero.sum()}): "
          f"mean A_map={A_mk_map[is_zero].mean():.3f} mK  "
          f"mean A_detected={A_mk_detected[is_zero].mean():.3f} mK  "
          f"false-detections={( ~detected_as_zero[is_zero]).sum()}\n")

    save_path = OUTPUT_DIR / f'random_recovery_{fg_model}.npz'
    np.savez(
        save_path,
        A_mk_true        = A_mk_true,
        nu0_true         = nu0_true,
        sigma_true       = sigma_true,
        is_zero          = is_zero,
        A_mk_map         = A_mk_map,
        A_mk_detected    = A_mk_detected,
        nu0_map          = nu0_map,
        sigma_map        = sigma_map,
        sigma_A_eff      = sigma_A_eff,
        detection_snr    = detection_snr,
        fg_params_map    = fg_map,
        bias_A           = bias_A,
        bias_nu0         = bias_nu0,
        bias_sigma       = bias_sigma,
        residuals_rms    = res_rms,
        success          = success,
        nu               = nu,
        nu_ref           = nu_ref,
        sigma_noise      = sigma_noise,
        fg_oracle        = FG_ORACLE[fg_model],
        t_int_hours      = T_INT_HOURS,
        delta_nu_mhz     = DELTA_NU_MHZ,
        detection_threshold = DETECTION_THRESHOLD,
    )
    print(f"  Saved → {save_path}")
