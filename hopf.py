import numpy as np
import cmath
from math import tau,pi
from SU2 import SU2_from_r3_sphere_point
from plane_torision import torsion_angle
from projection import plane_to_sphere

angles = np.linspace(0, tau, 360)
def norm_s(z:complex) -> float: return abs(z) / np.sqrt(1.0 + pow(abs(z),2))
def norm_c(z:complex) -> float: return 1.0 / np.sqrt(1.0 + pow(abs(z),2))


def s3_circle_pt(s2):
    """
    Return the circle on S^3 ⊂ C^2
    corresponding to the line through the origin in C^2 with slope a = tan(ϕ/2) e^iθ.
    where θ is the polar angle, and ϕ is the azimuthal angle of a spherical coordinate.
    This is the intersection of the complex line with the 3-sphere.
    """
    r,theta,phi = s2
    slope = np.tan(phi / 2) * cmath.exp(1j * theta)
    m = abs(slope)
    s = norm_s(slope)
    c = norm_c(slope)
    phase_a = slope / m if m != 0 else 1  # unit complex number
    z1 = c * cmath.exp(1j)
    z2 = s * phase_a * cmath.exp(1j)
    return z1,z2


#circle fiber in C2
def base_fiber(ts=angles):
  return np.array([
      [cmath.exp(1j*t), 0]
      for t in ts
  ],dtype=complex)

base_fiber = base_fiber()

def twisted_fiber(U,twist_angle,fiber=base_fiber):
  P = np.array([[np.exp(1j*twist_angle), 0],[0, np.exp(-1j*twist_angle)]],dtype=complex)
  return (U @ P @ fiber.T).T


def hopf_fiber_from_plane_curve_pt(curve_pts,index,R=1):
  p1, p2 = curve_pts[index - 1], curve_pts[index]
  ta = torsion_angle(p1, p2)
  sx,sy,sz = plane_to_sphere(p2,R)
  U = SU2_from_r3_sphere_point(sx,sy,sz)
  fiber = twisted_fiber(U,ta)
  return (U @ fiber.T).T

def torus_circle(c, s, u, v):
  """
  Vectorized torus stereographic projection.
  u and v must broadcast to the same shape.
  """
  u = np.asarray(u)
  v = np.asarray(v)
  # denominator
  d = (1 - s) * np.sin(v)
  d = np.where(np.abs(d) < 1e-9, np.sign(d) * 1e-9, d)
  x = (c * np.cos(u)) / d
  y = (c * np.sin(u)) / d
  z = (s * np.cos(v)) / d
  return np.stack((x, y, z), axis=-1)


def torus_circles_from_plane_curve_point(curve_pts,index,alpha,R=1):
  fiber = hopf_fiber_from_plane_curve_pt(curve_pts,index,R)
  # Torus radii (constant along the fiber)
  z1, z2 = fiber[0]
  c, s = abs(z1), abs(z2)
  u = angles
  #meridian torus circle
  mr_circ   = torus_circle(c, s, alpha, u)
  #paralell torus circle
  pr_circ   = torus_circle(c, s, u, alpha)
  #villarceau torus circle
  vil_circ  = torus_circle(c, s, u, u + alpha)
  #mirrored villarceau torus circle
  vil_circ2 = torus_circle(c, s, u, -u + alpha)
  return [mr_circ, pr_circ, vil_circ, vil_circ2]

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
