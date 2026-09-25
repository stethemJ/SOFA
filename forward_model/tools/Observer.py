import numpy as np
import numpy.typing as npt
import healpy as hp
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
import astropy.units as u


SIDEREAL_DAY_HOURS = 23.9344696  # mean sidereal day, in hours


class Observer:
    """A location + time on Earth, used to rotate a Galactic-frame sky
    map into the local horizontal frame and mask below-horizon pixels to
    a fixed ground temperature."""

    def __init__(self, latitude: float, longitude: float, elevation: float,
                 time, ground_temperature: float = 300.0):
        self.location = EarthLocation(lat=latitude * u.deg, lon=longitude * u.deg,
                                       height=elevation * u.m)
        self.time = Time(time)
        self.ground_temperature = ground_temperature
        self._grid_cache = {}  # nside -> (theta_gal, phi_gal, above_horizon)

    def _get_galactic_mapping(self, nside: int):
        if nside not in self._grid_cache:
            npix = hp.nside2npix(nside)
            theta_local, phi_local = hp.pix2ang(nside, np.arange(npix))
            alt = 90 - np.degrees(theta_local)
            az = np.degrees(phi_local)
            altaz = AltAz(alt=alt * u.deg, az=az * u.deg, location=self.location, obstime=self.time)
            gal = SkyCoord(altaz).galactic
            theta_gal = np.radians(90 - gal.b.deg)
            phi_gal = np.radians(gal.l.deg)
            self._grid_cache[nside] = (theta_gal, phi_gal, alt >= 0)
        return self._grid_cache[nside]

    def local_sky(self, sky_map: npt.ArrayLike, nside: int) -> npt.NDArray[np.float64]:
        """Downgrade `sky_map` to `nside`, rotate into this observer's
        local horizontal frame, and set below-horizon pixels to
        self.ground_temperature."""
        sky_map = hp.ud_grade(np.asarray(sky_map, dtype=float), nside)
        theta_gal, phi_gal, above_horizon = self._get_galactic_mapping(nside)
        local_map = hp.get_interp_val(sky_map, theta_gal, phi_gal)
        return np.where(above_horizon, local_map, self.ground_temperature)

    @classmethod
    def lst_scan(cls, latitude: float, longitude: float, elevation: float, start_time,
                 n_blocks: int = 24, ground_temperature: float = 300.0) -> list["Observer"]:
        """n_blocks Observer instances evenly spaced across one sidereal
        day starting at start_time -- an "LST scan" in hour blocks."""
        start = Time(start_time)
        step_hours = SIDEREAL_DAY_HOURS / n_blocks
        return [cls(latitude, longitude, elevation, start + i * step_hours * u.hour, ground_temperature)
                for i in range(n_blocks)]
