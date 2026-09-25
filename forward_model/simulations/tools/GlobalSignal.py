"""Physically-simulated 21-cm global (sky-averaged) signal via ARES
(Accelerated Reionization Era Simulations, github.com/mirochaj/ares).

NOTE: this module requires the `ares` package, which needs Python <=3.11
(it still uses the `imp` module, removed in Python 3.12) and a handful
of pinned dependency versions (see simulations/requirements.txt or requirements
below) to work with modern numpy/scipy/matplotlib. It is NOT installed
in the main project venv (env/) -- run anything that imports this
module with the dedicated interpreter at simulations/.venv311/bin/python3.
"""
import csv
import numpy as np
import numpy.typing as npt
import ares

REST_FREQ_MHZ = 1420.405751  # 21-cm hyperfine transition rest frequency

# Cosmological parameters: ARES's bundled Planck Legacy Archive chains,
# base LambdaCDM, TT+TE+EE+lowE combination (loaded by default; listed
# explicitly here for documentation/reproducibility).
_PLANCK_COSMOLOGY = {
    "cosmology_name": "planck_TTTEEE_lowl_lowE",
    "cosmology_id": "best",
}

# Astrophysical parameters: ARES's "simple" (fcoll-based) source model
# defaults (fstar=0.1, Tmin=1e4 K) with fX tuned down from its default
# of 1.0 to 0.07 -- this shifts the absorption trough from ~68 MHz
# (fX=1.0) to ~78 MHz, matching the EDGES low-band trough's location
# (Bowman et al. 2018). Standard astrophysics cannot reproduce EDGES'
# full ~-500 mK depth without exotic physics (Mirocha & Furlanetto
# 2019); this fiducial model reproduces the *location* honestly, with
# a physically self-consistent (shallower, ~-180 mK) depth -- found by
# a direct scan (see simulations/scripts/generate_signal.py's printed output),
# not read off a specific published parameter table.
FIDUCIAL_PARAMS = {
    **_PLANCK_COSMOLOGY,
    "fX": 0.07,
    "fstar": 0.1,
    "Tmin": 1e4,
}


class GlobalSignal:
    """Thin wrapper around ares.simulations.Global21cm."""

    def __init__(self, **overrides):
        self.params = {**FIDUCIAL_PARAMS, **overrides}
        self.sim = None

    def run(self) -> "GlobalSignal":
        self.sim = ares.simulations.Global21cm(**self.params)
        self.sim.run()
        return self

    def _history(self):
        if self.sim is None:
            raise RuntimeError("Call run() before accessing results.")
        return self.sim.history

    def to_records(self) -> list[dict]:
        """Rows in the same schema as data/injected_21cm_signal.csv:
        redshift, frequency_MHz, brightness_temp_mK, neutral_fraction.
        Sorted by ascending frequency."""
        h = self._history()
        z = np.asarray(h["z"], dtype=float)
        freq_mhz = REST_FREQ_MHZ / (1 + z)
        order = np.argsort(freq_mhz)
        z, freq_mhz = z[order], freq_mhz[order]
        dTb_mk = np.asarray(h["dTb"], dtype=float)[order]
        x_hi = np.asarray(h["igm_h_1"], dtype=float)[order]
        return [
            {"redshift": zi, "frequency_MHz": fi, "brightness_temp_mK": ti, "neutral_fraction": xi}
            for zi, fi, ti, xi in zip(z, freq_mhz, dTb_mk, x_hi)
        ]

    def export_csv(self, path) -> None:
        records = self.to_records()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["redshift", "frequency_MHz", "brightness_temp_mK", "neutral_fraction"])
            writer.writeheader()
            writer.writerows(records)

    def trough(self) -> tuple[float, float, float]:
        """Returns (frequency_MHz, redshift, brightness_temp_mK) at the
        global minimum of the brightness temperature."""
        h = self._history()
        dTb = np.asarray(h["dTb"], dtype=float)
        i = int(np.argmin(dTb))
        z = float(h["z"][i])
        return REST_FREQ_MHZ / (1 + z), z, float(dTb[i])
