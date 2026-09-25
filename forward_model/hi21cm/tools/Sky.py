import csv
import numpy as np
import numpy.typing as npt

from forward_model.tools.config import PROJECT_ROOT
from forward_model.foregrounds.tools.config import TE_MAP_PATH, EM_MAP_PATH
from forward_model.foregrounds.tools.Sky import Sky as _ForegroundSky

CSV_PATH = PROJECT_ROOT / "data" / "ares_21cm_signal.csv"


class Sky(_ForegroundSky):
    """Spectral-index foreground sky plus an injected isotropic 21-cm global-signal monopole."""

    def __init__(self, csv_path=CSV_PATH, nside: int = 64,
                 te_map_path=TE_MAP_PATH, em_map_path=EM_MAP_PATH):
        super().__init__(nside=nside, te_map_path=te_map_path, em_map_path=em_map_path)
        self._freq_mhz, self._temp_k = self._load_21cm_csv(csv_path)

    @staticmethod
    def _load_21cm_csv(csv_path):
        with open(csv_path, newline="") as f:
            rows = [(float(r["frequency_MHz"]), float(r["brightness_temp_mK"]))
                     for r in csv.DictReader(f)]
        freq_mhz, temp_mk = map(np.array, zip(*rows))
        order = np.argsort(freq_mhz)
        return freq_mhz[order], temp_mk[order] * 1e-3  # mK -> K

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
