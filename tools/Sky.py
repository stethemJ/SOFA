import csv
import numpy as np
import numpy.typing as npt
from pygdsm import GlobalSkyModel

from tools.config import PROJECT_ROOT

CSV_PATH = PROJECT_ROOT / "data" / "injected_21cm_signal.csv"


class Sky:
    """GSM2008 diffuse Galactic foreground (pygdsm) plus an injected
    isotropic 21-cm global-signal monopole."""

    def __init__(self, csv_path=CSV_PATH, **gsm_kwargs):
        gsm_kwargs.setdefault("freq_unit", "MHz")
        self._gsm = GlobalSkyModel(**gsm_kwargs)  # output units: K (fixed for GSM2008)
        self._freq_mhz, self._temp_k = self._load_21cm_csv(csv_path)  # ascending freq

    @staticmethod
    def _load_21cm_csv(csv_path):
        with open(csv_path, newline="") as f:
            rows = [(float(r["frequency_MHz"]), float(r["brightness_temp_mK"]))
                     for r in csv.DictReader(f)]
        freq_mhz, temp_mk = map(np.array, zip(*rows))
        order = np.argsort(freq_mhz)  # CSV is descending in frequency
        return freq_mhz[order], temp_mk[order] * 1e-3  # mK -> K

    def generate(self, frequency: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Raw GSM2008 healpix map(s) (K) at frequency/frequencies (MHz)."""
        try:
            return self._gsm.generate(frequency)
        except RuntimeError as e:
            raise ValueError(f"GSM2008 requires 10 MHz <= frequency <= 94000 MHz: {e}") from e

    def _interp_21cm_k(self, frequency: npt.ArrayLike) -> npt.NDArray[np.float64]:
        frequency = np.atleast_1d(np.asarray(frequency, dtype=float))
        lo, hi = self._freq_mhz[0], self._freq_mhz[-1]
        if np.any(frequency < lo) or np.any(frequency > hi):
            raise ValueError(f"frequency outside tabulated 21cm range [{lo:.3f}, {hi:.3f}] MHz")
        return np.interp(frequency, self._freq_mhz, self._temp_k)

    def inject_21cm(self, frequency: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """generate(frequency) plus a uniform interpolated 21-cm offset (K)."""
        sky = self.generate(frequency)
        offset = self._interp_21cm_k(frequency)
        return sky + offset[0] if sky.ndim == 1 else sky + offset[:, np.newaxis]
