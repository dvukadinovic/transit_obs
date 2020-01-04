"""
@author = Dusan Vukadinovic
@vesion = v0.1
@date   = 2019-12-27

mini skriptica koja vraca dane u kojima je pogodno posmatrati dati tranzit.

pogodni dani su oni kada je osvetljenost Meseca ispod 50%, kada se objekat koji posmatramo nalazi 30 stepeni iznad horizonta (kada je maximum tranzita) i kada se tranzit desava dok je na datoj lokaciji noc. Predvidjeni datumi su za narednih godinu dana.

to add:
  1. napraviti classu za posmatrani objekat
  2. napraviti classu za mesto posmatranja
  4. odabrati one tranzitne dane kojih se u celosti vide date posmatracke noci (tako da su pocetak i kraj +/- jos koji ~15min) vidljivi date noci
  5. napraviti da radi za listu objekata (tranzita) i za svaki da vrati potrebne informacije (?)
  6. za pogodne dane napraviti grafik promene visine i pozicije meseca; oznaciti i pocetak, sredinu i kraj tranzita (?)
  7. napraviti module od ovoga
  8. dodati da se podaci o planetama preuzimaju sa odredjenog sajta o exoplnetama

dnevnik:

4.1.2020: dopisao opise funkcija i ulazne/izlazne parametre istih

"""

from datetime import datetime, timedelta
import numpy as np
import numpy.ma as ma
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, get_sun, get_moon
from astropy.time import Time
import matplotlib.pyplot as plt

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
    """
    i = moon_phase_angle(time, ephemeris=ephemeris)
    k = (1 + np.cos(i))/2.0
    return k.value

def cycle_or_not(time):
	"""
	From input 'time' return when is the illumination of the moon belove 0.5.

	If the phase is less then a 0.5 (time at which Moon reaches first quater) we are having an observational cycle. If it is greater than 0.5, we are out of observational cycle.

	Parameters
	----------
	time : '~datetime.datetime'
		Time for which to determine if moon illumination is less then 0.5.

	Returns
	-------
	time : '~datetime.datetime'
		Time for which we have illumination less then 0.5 (empty if condition is not met).
	"""
	time = Time(time)
	illumination = moon_illumination(time)
	mask = ma.masked_less_equal(illumination, 0.5).mask
	return time[mask]

def lst(loc_time, loc):
	"""
	Calculate the Local Sidereal Time for given location on the Earth and local time. 

	Ref: http://www.jgiesen.de/astro/astroJS/siderealClock/

	Parameters
	----------
	loc_time : '~datetime.datetime'
		Times for which to calculate LST.
	loc : 'special dict' (with objects 'loc','lat','tz')
		Observer location on Earth. Currently special dict object.

	Returns
	-------
	LMST : float
		Local mean sidereal time [hours].
	"""
	MJD = Time(loc_time).mjd
	MJD0 = np.floor(MJD)
	ut = (MJD - MJD0)*24.0
	t_eph = (MJD0-51544.5)/36525.0
	GMST = 6.697374558 + 1.0027379093*ut + (8640184.812866 + (0.093104 - 0.0000062*t_eph)*t_eph)*t_eph/3600.0
  	
	LMST = 24.0*np.modf((GMST + loc["lon"]/15.0)/24.0)[0] - loc["tz"]
	return LMST

def eq2hor(obj, loc, time):
	"""
	Conversion from equatorial to horizontal coordinates.
	
	Parameters
	----------
	obj : 'special dict' (with objects 'ra', 'dec', 'T0', 'P', 'duration')
		Dictionary with RA and DEC of object in degrees.
	loc : 'special dict' (with objects 'lon', 'lat', 'tz')
		Observatory loc on Earth and time zone.
	time : '~datetime.datetime'
		Value (or array) of local time for which conversion should be done.

	Returns
	-------
	hegight : ndarray
		Height of object [degrees].
	A : ndarray
		Azimuth of object [degrees].
	"""
	ra, dec = obj["ra"], obj["dec"]
	phi = np.deg2rad(loc["lat"])
	dec = np.deg2rad(dec)
	
	h = np.deg2rad(lst(time,loc)*15 - ra)
	
	x = -np.sin(phi)*np.cos(dec)*np.cos(h) + np.cos(phi)*np.sin(dec)
	y = np.cos(dec)*np.sin(h)
	A = -np.arctan2(y,x)	# azimuth = -arctan(y/x)
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
	From the input parameters of given transit object 'obj' determine when on given location 'loc' we have transit in next 366 days.
	
	Parameters
	----------
	obj : 'special dict' (with objects 'ra', 'dec', 'T0', 'P', 'duration')
		Dictionary with RA and DEC of object in degrees.
	loc : 'special dict' (with objects 'lon', 'lat', 'tz')
		Observatory loc on Earth and time zone.

	Returns
	-------
	time : '~datetime.datetime'
		Times of transit mid-point in next 366 days.
	"""
	epoch = obj["T0"]
	period = obj["P"]
	duration = obj["duration"]

	now_loc = datetime.now()
	now_utc = now_loc - timedelta(hours=loc["tz"])
	now_jd = Time(now_utc).jd
	phase = np.modf((now_jd - epoch)/period)[0] # faza tranzita; 0 je mid tranzita
	next_transit = now_loc + timedelta(days=(1-phase)*period)

	# trenuci mid-pointa tranzita u narednih godinu dana
	# po lokalnom vremenu (bez uracunatog letenjg/zimskog racunanja vremena)
	time = np.arange(next_transit, next_transit+timedelta(days=366), timedelta(days=period), dtype='datetime64').astype(datetime)

	return time

def main(obj, loc, full_output=False):
	"""
	For given object 'obj' and location on the Earth 'loc' calculate on which days during next 366 days we can observe given transit object.

	Conditions for days of interest are:
		- moon illumination is less then 0.5 (observational cycle)
		- mid-point of transit is during night time at height above 30 degrees

	Parameters
	----------
	obj : 'special dict' (with objects 'ra', 'dec', 'T0', 'P', 'duration')
		Dictionary with RA and DEC of object in degrees.
	loc : 'special dict' (with objects 'lon', 'lat', 'tz')
		Observatory loc on Earth and time zone.
	full_output : bool, optional
		If False, then only suitable days for observation are return. If True, full ouput is returned: suitable days with time of mid-point of transit and object height.

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
	sun = {"ra" : sun.ra.value, "dec" : sun.dec.value}
	sun_alt, az = eq2hor(sun, loc, above_horizon_time)
	mask = ma.masked_less_equal(sun_alt, -18).mask
	above_and_night_time = above_horizon_time[mask]
	sun_alt = sun_alt[mask]
	alt = alt[mask]

	print(above_and_night_time)
	print(sun_alt)
	print(alt)

#=-- determine "obj" height on given "obs_semesters" and for given "loc"
loc = {"lon":21.35,"lat":43.1402,"tz":+0}			# Vidojevica Observatory
obj = {"ra":(37.41/3600 + 50/60 + 13)*15, "dec":-(14.4/3600 + 48/60 + 6),
		"T0": 2455624.26679, "P":1.3371182, "duration":108.6}
# epoch  = time of minimum of transit [JD]
# period = transit period [d]
# duration = duration of transit [min]

main(obj, loc)