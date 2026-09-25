import numpy as np
import numpy.typing as npt


class GaussianSignal:
    """Isotropic Gaussian absorption feature injected uniformly across all sky pixels.

    Models a global 21-cm-like signal: a single Gaussian trough in frequency space,
    broadcast as a monopole so every pixel receives the same offset.

    Parameters
    ----------
    A_mk : float
        Absorption depth [milli-Kelvin]. Always positive; the feature is
        subtracted from the sky (absorption convention).
    nu0 : float
        Centre frequency [MHz].
    sigma : float
        Gaussian width [MHz] (1-sigma). FWHM = 2.355 * sigma.
    """

    def __init__(self, A_mk: float = 200.0, nu0: float = 78.0, sigma: float = 10.0):
        if A_mk < 0:
            raise ValueError("A_mk must be non-negative (absorption depth)")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.A_mk = A_mk
        self.nu0 = nu0
        self.sigma = sigma

    def temperature(self, nu: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Gaussian signal brightness temperature [K] at frequency/frequencies nu [MHz].

        Returns a negative array (absorption below zero).
        """
        nu = np.asarray(nu, dtype=float)
        return -(self.A_mk * 1e-3) * np.exp(-0.5 * ((nu - self.nu0) / self.sigma) ** 2)

    def inject(self, sky_maps: npt.NDArray[np.float64],
               freqs_mhz: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Add signal as a uniform offset to foreground HEALPix maps.

        Parameters
        ----------
        sky_maps : ndarray, shape (npix,) or (nfreq, npix)
        freqs_mhz : scalar or array of length nfreq [MHz]

        Returns
        -------
        ndarray, same shape as sky_maps
        """
        freqs = np.atleast_1d(np.asarray(freqs_mhz, dtype=float))
        T_sig = self.temperature(freqs)
        if sky_maps.ndim == 1:
            return sky_maps + float(T_sig[0])
        return sky_maps + T_sig[:, np.newaxis]
