import numpy as np
import numpy.typing as npt

from forward_model.hi21cm.tools.Sky import Sky
from antenna.tools.EField import EField
from antenna.tools.VshBasis import VshBasis
from forward_model.tools.Observer import Observer
from forward_model.tools import Convolution


class Analyze:
    """Thin facade: composes Sky + Observer + EField + Convolution for a
    given beam, height, and observer -- no physics logic of its own."""

    def __init__(self, basis: VshBasis, height: float, observer: Observer,
                 l_max: int = 40, nside: int = 128, inject_21cm: bool = False):
        self.basis = basis
        self.height = height
        self.observer = observer
        self.sky = Sky(nside=nside)
        self.l_max = l_max
        self.nside = nside
        self.inject_21cm = inject_21cm

    def _generate_sky(self, freqs_mhz):
        return self.sky.inject_21cm(freqs_mhz) if self.inject_21cm else self.sky.generate(freqs_mhz)

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
