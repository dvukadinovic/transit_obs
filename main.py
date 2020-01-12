"""
@author = Dusan Vukadinovic
@vesion = v0.1
@date   = 2019-12-27

mini skriptica koja vraca dane u kojima je pogodno posmatrati dati tranzit.

pogodni dani su oni kada je osvetljenost Meseca ispod 50%, kada se objekat 
koji posmatramo nalazi 30 stepeni iznad horizonta (kada je maximum tranzita) 
i kada se tranzit desava dok je na datoj lokaciji noc. Predvidjeni datumi su 
za narednih godinu dana.

to add:
  4. odabrati one tranzitne dane kojih se u celosti vide date posmatracke 
     noci (tako da su pocetak i kraj +/- jos koji ~15min) vidljivi date noci
  5. napraviti da radi za listu objekata (tranzita) i za svaki da vrati 
     potrebne informacije (?)
  6. za pogodne dane napraviti grafik promene visine i pozicije meseca; 
     oznaciti i pocetak, sredinu i kraj tranzita (?)
  7. napraviti module od ovoga
  8. dodati da se podaci o planetama preuzimaju sa odredjenog sajta o 
     exoplnetama

dnevnik:

04.01.2020: dopisao opise funkcija i ulazne/izlazne parametre istih
12.01.2020: napravio klase za opservatoriju i objekat. 'Opservatorija' 
            nasledjuje klasu '~astropy.coordinates.EarthLocation' s 
            tim da je dodata instanca 'time_zone'. 'Objekat' nasledjuje
            klasu '~astropy.coordinates.SkyCoord' i dodate su instance koje
            se odnose na tranzit 'T0', 'period' i 'duration'.
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
from datetime import datetime, timedelta
from astropy.coordinates import EarthLocation, SkyCoord, get_sun, get_moon
from astropy.time import Time
import astropy.units as u
from scipy.constants import speed_of_light, astronomical_unit

class Observatory(EarthLocation):
    """
    Class for defining location (latitude and longitude) of observatory site.
    It is inherited from 'astropy.coordinates.EarthLocation' with added 
    instance of time_zone.
    """
    def __init__(self, lon, lat, tz):
        super().__init__()
        self.time_zone = tz

class Object(SkyCoord):
    """
    Class that creates transit object with given coordinates in RA and DEC. 
    Class is inherited from '~astropy.coordinates.SkyCoord' with additional
    instances

    Additional instances
    --------------------
    T0 : float
        Epoch of transit-mid point (in JD).
    period : float
        Transit period (in days).
    duration : float
        Transit duration (in minutes).
    """
    def __init__(self, ra, dec, units="deg", T0=None, period=None, duration=None):
        super(Object, self).__init__(ra=ra, dec=dec, unit=units)
        self.T0 = T0
        self.period = period
        self.duration = duration

def jd2hjd(time, obj):
    """
    Converting from HJD to JD.

    Parameters
    ----------
    time : '~astropy.time.Time'
        Time of interest.
    obj : '~Object'
        Transit object which we observe.

    Returns
    -------
    hjd : ndarray
        Julian day.
    """
    jd = time.jd
    sun = get_sun(time)
    ra_sun = np.deg2rad(sun.ra.value)
    dec_sun = np.deg2rad(sun.dec.value)

    ra = np.deg2rad(obj.ra.value)
    dec = np.deg2rad(obj.dec.value)
    
    hjd = jd - sun.distance.value*astronomical_unit/speed_of_light \
                * (np.sin(dec)*np.sin(dec_sun) \
                    + np.cos(dec)*np.cos(dec_sun)*np.cos(ra-ra_sun))
    return hjd

def moon_phase_angle(time, ephemeris=None):
    """
    Calculate lunar orbital phase in radians.

    Parameters
    ----------
    time : `~astropy.time.Time`
        Time of observation
    ephemeris : str, optional
        Ephemeris to use.  If not given, use the one set with
        `~astropy.coordinates.solar_system_ephemeris` (which is
        set to 'builtin' by default).

    Returns
    -------
    i : float
        Phase angle of the moon [radians]

    References
    ----------
    https://github.com/astropy/astroplan
    """
    sun = get_sun(time)
    moon = get_moon(time, ephemeris=ephemeris)
    elongation = sun.separation(moon)
    return np.arctan2(sun.distance*np.sin(elongation),
                      moon.distance - sun.distance*np.cos(elongation))

def moon_illumination(time, ephemeris=None):
    """
    Calculate fraction of the moon illuminated.

    Parameters
    ----------
    time : `~astropy.time.Time`
        Time of observation
    ephemeris : str, optional
        Ephemeris to use.  If not given, use the one set with
        `~astropy.coordinates.solar_system_ephemeris` (which is
        set to 'builtin' by default).

    Returns
    -------
    k : float
        Fraction of moon illuminated

    References
    ----------
    https://github.com/astropy/astroplan
    """
    i = moon_phase_angle(time, ephemeris=ephemeris)
    k = (1 + np.cos(i))/2.0
    return k.value

def cycle_or_not(time):
    """
    From input 'time' return when is the illumination of the moon belove 0.5.

    If the phase is less then a 0.5 (time at which Moon reaches first quater) 
    we are having an observational cycle. If it is greater than 0.5, we are 
    out of observational cycle.

    Parameters
    ----------
    time : '~datetime.datetime'
        Time for which to determine if moon illumination is less then 0.5.

    Returns
    -------
    time : '~datetime.datetime'
        Time for which we have illumination less then 0.5 (empty if condition 
        is not met).
    """
    time = Time(time)
    illumination = moon_illumination(time)
    mask = ma.masked_less_equal(illumination, 0.5).mask
    return time[mask]

def lst(loc_time, loc):
    """
    Calculate the Local Sidereal Time for given location on the Earth and 
    local time. 

    Parameters
    ----------
    loc_time : '~datetime.datetime'
        Times for which to calculate LST.
    loc : 'Observatory'
        Observer location on Earth.

    Returns
    -------
    LMST : float
        Local mean sidereal time [hours].

    References
    ----------
    http://www.jgiesen.de/astro/astroJS/siderealClock/
    """
    MJD = Time(loc_time).mjd
    MJD0 = np.floor(MJD)
    ut = (MJD - MJD0)*24.0
    t_eph = (MJD0-51544.5)/36525.0
    GMST = 6.697374558 + 1.0027379093*ut + (8640184.812866 + (0.093104 - 0.0000062*t_eph)*t_eph)*t_eph/3600.0
    
    LMST = 24.0*np.modf((GMST + loc.lon.value/15.0)/24.0)[0] - loc.time_zone
    return LMST

def eq2hor(obj, loc, time):
    """
    Conversion from equatorial to horizontal coordinates.
    
    Parameters
    ----------
    obj : '~Object'
        Transit object.
    loc : '~Observatory'
        Observatory location on Earth and time zone.
    time : '~datetime.datetime'
        Value (or array) of local time for which conversion should be done.

    Returns
    -------
    hegight : ndarray
        Height of object [degrees].
    A : ndarray
        Azimuth of object [degrees].
    """
    ra, dec = obj.ra.value, obj.dec.value
    phi = np.deg2rad(loc.lat.value)
    dec = np.deg2rad(dec)
    
    h = np.deg2rad(lst(time,loc)*15 - ra)
    
    x = -np.sin(phi)*np.cos(dec)*np.cos(h) + np.cos(phi)*np.sin(dec)
    y = np.cos(dec)*np.sin(h)
    A = -np.arctan2(y,x)    # azimuth = -arctan(y/x)
    # correction for negative values of azimuth
    if type(A) is np.ndarray:
        for ii in range(len(A)):
            if A[ii] < 0:
                A[ii] += 2*np.pi
    else:
        if A<0:
            A += 2*np.pi
    height = np.arcsin( np.sin(phi)*np.sin(dec) + np.cos(phi)*np.cos(dec)*np.cos(h) )
    height = np.rad2deg(height)

    mask = ma.masked_less_equal(height, 30).mask    

    return height, np.rad2deg(A)

def future_transits(obj, loc):
    """
    From the input parameters of given transit object 'obj' determine when on 
    given location 'loc' we have transit in next 366 days.
    
    Parameters
    ----------
    obj : '~Object'
        Transit object.
    loc : '~Observatory'
        Observatory location on Earth. 

    Returns
    -------
    time : '~datetime.datetime'
        Times of transit mid-point in next 366 days.
    """
    epoch = obj.T0
    period = obj.period
    duration = obj.duration

    now_loc = datetime.now()
    now_utc = now_loc - timedelta(hours=loc.time_zone)
    now_hjd = jd2hjd(Time(now_utc), obj)
    phase = np.modf((now_hjd - epoch)/period)[0] # faza tranzita; 0 je mid tranzita
    next_transit = now_loc + timedelta(days=(1-phase)*period)

    # trenuci mid-pointa tranzita u narednih godinu dana
    # po lokalnom vremenu (bez uracunatog letenjg/zimskog racunanja vremena)
    time = np.arange(next_transit, next_transit+timedelta(days=366), timedelta(days=period), dtype='datetime64').astype(datetime)

    return time

def main(obj, loc, full_output=False):
    """ 
    For given object 'obj' and location on the Earth 'loc' calculate on which
    days during next 366 days we can observe given transit object.

    Conditions for days of interest are:
        - moon illumination is less then 0.5 (observational cycle)
        - mid-point of transit is during night time at height above 30 degrees

    Parameters
    ----------
    obj : '~Object'
        Transit object.
    loc : '~Observatory'
        Observatory location on Earth.
    full_output : bool, optional
        If False, then only suitable days for observation are return. If True,
        full ouput is returned: suitable days with time of mid-point of
        transit and object height.

    Returns
    -------
    days : '~datetime.datetime'
        Suitable days for transit observation.
    obj_alt : ndarray
        Object height on suitable days (if 'full_output' is True).
    """
    # trenuci sredine tranzita u narednih godinu dana
    mid_of_transit = future_transits(obj, loc)
    # trenuci tranzita koji se padaju kada je mesec osvetljen manje od 50%
    obs_mid_of_transit = cycle_or_not(mid_of_transit)
    
    # trenuci kada je tranzit i kada je objekat 30deg iznad horizonta
    alt, az = eq2hor(obj, loc, obs_mid_of_transit)
    mask = ma.masked_greater_equal(alt, 30).mask
    above_horizon_time = obs_mid_of_transit[mask]
    alt = alt[mask]

    # trenuci kada je tranzit tokom noci
    sun = get_sun(above_horizon_time)
    sun_alt, az = eq2hor(sun, loc, above_horizon_time)
    mask = ma.masked_less_equal(sun_alt, -18).mask
    above_and_night_time = above_horizon_time[mask]
    sun_alt = sun_alt[mask]
    alt = alt[mask]

    print(above_and_night_time)
    print(sun_alt)
    print(alt)

# Vidojevica Observatory
loc = Observatory(21.35, 43.1402, 1)
obj = Object(ra=13.843725, dec=-6.804, T0=2455624.26679, 
             period=1.3371182, duration=108.6, units=("hour","deg"))

main(obj, loc)