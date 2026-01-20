import numpy as np
import cmath

# Build (z1,z2) for a Hopf coordinate (theta,phi,t)
def hopf_zpair(theta: float, phi: float, t: float):
    """
    Returns complex pair (z1, z2) on S^3 corresponding to Hopf coords:
      slope a = tan(phi/2) * e^{i theta}
      z1 = c * e^{i t}
      z2 = s * phase_a * e^{i t}
    """
    slope = cmath.exp(1j * theta) * np.tan(phi / 2.0)
    mag = abs(slope)
    s = mag / np.sqrt(1.0 + mag*mag)
    c = 1.0 / np.sqrt(1.0 + mag*mag)
    phase_a = slope / mag if mag != 0.0 else 1.0 + 0j
    e_it = cmath.exp(1j * t)
    z1 = c * e_it
    z2 = s * phase_a * e_it
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