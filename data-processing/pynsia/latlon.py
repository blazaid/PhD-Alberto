from math import sin, cos, sqrt, atan2, radians

EARTH_RADIUS = 6373.0

def distance(c1, c2, degrees=True):
    """
    :param c1: The first coordinate as a tuple either in the form (lat, lon)
        or (lon, lat).
    :param c2: The second coordinate as a tuple in the same form as c1.
    :param degrees: If True, the coordinates are suposed to be in degrees. If
        not, the coordinates are suposed to be in radians.
    :return: The distance between the two coordinates in meters.
    """
    (lat1, lon1), (lat2, lon2) = c1, c2
    if degrees:
        lat1 = radians(lat1)
        lon1 = radians(lon1)
        lat2 = radians(lat2)
        lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2.) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return EARTH_RADIUS * c * 1000.0
