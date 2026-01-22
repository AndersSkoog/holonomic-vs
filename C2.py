import numpy as np
import cmath

def hopf_zpair(theta: float, phi: float, t: float):
    """
    Returns (z1,z2) on S^3 given Hopf coords (theta,phi,t).
    Safe for all phi in [0, pi].
    """
    e_it = cmath.exp(1j * t)
    z1 = cmath.cos(phi/2) * e_it
    z2 = cmath.sin(phi/2) * cmath.exp(1j*(t + theta))
    return z1, z2



# Convert complex pair (z1,z2) -> unit quaternion components (w,x,y,z)
# q = w + x i + y j + z k
def zpair_to_quaternion(z1: complex, z2: complex):
    w = z1.real
    x = z1.imag
    y = z2.real
    z = z2.imag
    # optionally renormalize to avoid numerical drift
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        return 1.0, 0.0, 0.0, 0.0
    return [w / norm, x / norm, y / norm, z / norm]