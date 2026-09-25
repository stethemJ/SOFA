import numpy as np
import numpy.typing as npt

from forward_model.foregrounds.tools.Sky import Sky as ForegroundSky
from forward_model.signal.tools.Signal import GaussianSignal
from antenna.tools.EField import EField
from antenna.tools.VshBasis import VshBasis
from forward_model.tools.Observer import Observer
from forward_model.tools import Convolution


class Analyze:
    """Thin facade: composes ForegroundSky + GaussianSignal + Observer + EField + Convolution
    for a given beam, height, and observer -- no physics logic of its own."""

    def __init__(self, basis: VshBasis, height: float, observer: Observer,
                 l_max: int = 40, nside: int = 128,
                 signal: GaussianSignal | None = None):
        self.basis = basis
        self.height = height
        self.observer = observer
        self.sky = ForegroundSky(nside=nside)
        self.signal = signal
        self.l_max = l_max
        self.nside = nside

    def _generate_sky(self, freqs_mhz):
        maps = self.sky.generate(freqs_mhz)
        if self.signal is not None:
            maps = self.signal.inject(maps, freqs_mhz)
        return maps

    def antenna_temperature(self, frequency_mhz: float) -> float:
        sky_map = self._generate_sky(frequency_mhz)
        local_map = self.observer.local_sky(sky_map, self.nside)
        sky_coeffs = Convolution.sky_coefficients(local_map, self.l_max, self.nside)
        ef = EField.from_basis(self.basis, frequency_mhz * 1e6, self.height)
        beam_coeffs = Convolution.beam_power_coefficients(ef, self.l_max, self.nside)
        return Convolution.antenna_temperature(beam_coeffs, sky_coeffs)

    def spectrum(self, freqs_mhz: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Convenience: antenna_temperature at each frequency, batching
        sky generation once."""
        freqs_mhz = np.asarray(freqs_mhz, dtype=float)
        sky_maps = np.atleast_2d(self._generate_sky(freqs_mhz))
        T_values = np.empty(len(freqs_mhz))
        for i, freq_mhz in enumerate(freqs_mhz):
            local_map = self.observer.local_sky(sky_maps[i], self.nside)
            sky_coeffs = Convolution.sky_coefficients(local_map, self.l_max, self.nside)
            ef = EField.from_basis(self.basis, freq_mhz * 1e6, self.height)
            beam_coeffs = Convolution.beam_power_coefficients(ef, self.l_max, self.nside)
            T_values[i] = Convolution.antenna_temperature(beam_coeffs, sky_coeffs)
        return T_values

    def lst_averaged_spectrum(self, freqs_mhz: npt.ArrayLike, observers: list[Observer]) -> npt.NDArray[np.float64]:
        """Average antenna_temperature(freq) across multiple observers
        (e.g. an LST scan). Batches sky generation once across all
        frequencies, and beam projection once per frequency (shared
        across observers, since the beam doesn't depend on the observer)."""
        freqs_mhz = np.asarray(freqs_mhz, dtype=float)
        sky_maps = np.atleast_2d(self._generate_sky(freqs_mhz))
        T_stack = np.empty((len(observers), len(freqs_mhz)))
        for i, freq_mhz in enumerate(freqs_mhz):
            ef = EField.from_basis(self.basis, freq_mhz * 1e6, self.height)
            beam_coeffs = Convolution.beam_power_coefficients(ef, self.l_max, self.nside)
            for j, observer in enumerate(observers):
                local_map = observer.local_sky(sky_maps[i], self.nside)
                sky_coeffs = Convolution.sky_coefficients(local_map, self.l_max, self.nside)
                T_stack[j, i] = Convolution.antenna_temperature(beam_coeffs, sky_coeffs)
        return T_stack.mean(axis=0)
