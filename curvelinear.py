import numpy as np
from lib import normalize_vector
from math import sin, cos, tan, atan2, acos
# -----------------------------
# Curvilinear rendering
# -----------------------------

def direction_to_angles(d: Sequence[float]) -> np.array:
    """Return (theta, phi) where theta = angle from forward axis (z), phi = atan2(y,x)."""
    d = normalize(np.array(d))
    theta = acos(max(-1.0, min(1.0, d[2])))
    phi = atan2(d[1], d[0])
    return np.array([theta, phi])

def fisheye_equidistant(p: Sequence[float], f: float = 1.0) -> np.array:
    theta, phi = direction_to_angles(p)
    r = f * theta
    return np.array([r * cos(phi), r * sin(phi)])

def fisheye_equisolid(p: Sequence[float], f: float = 1.0) -> np.array:
    theta, phi = direction_to_angles(p)
    r = 2.0 * f * sin(theta / 2.0)
    return np.array([r * cos(phi), r * sin(phi)])

def fisheye_stereographic(p: Sequence[float], f: float = 1.0) -> np.array:
    theta, phi = direction_to_angles(p)
    r = 2.0 * f * tan(theta / 2.0)
    return np.array([r * cos(phi), r * sin(phi)])

def orthographic_onto_disc(p: Sequence[float], f: float = 1.0) -> np.array:
    theta, phi = direction_to_angles(p)
    r = f * sin(theta)
    return np.array([r * cos(phi), r * sin(phi)])

def equirectangular(p: Sequence[float]) -> np.array:
    d = normalize(p)
    theta = acos(max(-1.0, min(1.0, d[2])))  # 0..pi
    phi = atan2(d[1], d[0])  # -pi..pi
    u = (phi + pi) / (2.0 * pi)
    v = 1.0 - (theta / pi)
    return np.array([u, v])
