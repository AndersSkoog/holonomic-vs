import numpy as np
from math import sin, cos, tan, acos, atan2, pi
from typing import Sequence, List


TAU = 2 * pi
MIN_VAL = 1e-9
fov = sin(pi / 5)
gamma = acos(4 / 5)
lambd = tan(gamma / 2)
npole = np.array([0.0, 0.0, 1.0])


def orthonormal_frame(S: np.ndarray):
  if abs(S[2]) < (1.0-MIN_VAL): ref = np.array([0.0, 0.0, 1.0])
  else: ref = np.array([1.0, 0.0, 0.0])
  u = np.cross(ref, S)
  u /= np.linalg.norm(u)
  v = np.cross(S, u)
  return u, v


def lift_to_sphere_in_direction(p: np.ndarray, S: np.ndarray) -> np.ndarray:
  u, v = orthonormal_frame(S)
  x, y = p
  r = x*x + y*y
  d = r + 1.0
  # point in local coordinates
  s_local = (2*x)*u + (2*y)*v + (r - 1) * S
  s = s_local / (r + 1)
  return s / np.linalg.norm(s)



def plane_to_sphere(p: np.ndarray) -> np.ndarray:
  x, y = p
  r = x*x + y*y
  d = r + 1.0
  sx = 2 * x / d
  sy = 2 * y / d
  sz = (r - 1) / d
  return np.array([sx, sy, sz])


# ---------- conversions ----------

def to_points(L: Sequence[complex]) -> List[np.ndarray]:
  return [np.array([p.real, p.imag]) for p in L]


def to_cmplx(L: Sequence[np.ndarray]) -> List[complex]:
  return [complex(p[0], p[1]) for p in L]


# ---------- Möbius transformatins from sphere point ----------

def s2pt_to_mobius_coef(s2pt: np.ndarray):
  N = np.array([0.0, 0.0, 1.0])
  S = np.asarray(s2pt)
  axis = np.cross(N, S)
  norm = np.linalg.norm(axis)
  if norm < MIN_VAL: return 1+0j, 0+0j, 0+0j, 1+0j
  axis = axis / norm
  dot = np.clip(np.dot(N, S), -1.0, 1.0)
  psi = np.arccos(dot)
  s = np.sin(psi / 2)
  w = np.cos(psi / 2)
  i, j, k = axis * s
  a = complex(w, i)
  b = complex(j, k)
  c = -np.conj(b)
  d = np.conj(a)
  return a, b, c, d


# ---------- Apply Möbius transform ----------

def apply_mobius(z: complex, coefs):
  a, b, c, d = coefs
  denom = c*z + d
  if abs(denom) < MIN_VAL: return complex(np.inf)
  return (a*z + b) / denom


def boundaryless_disc_transform(z: complex, coefs):
  z1 = apply_mobius(z, coefs)
  if abs(z1) <= 1: return z1
  return 1 / np.conj(z1)


# ---------- core holonomic rearrangement ----------

def holonomic_rearrangement(D: Sequence[np.ndarray], index: int) -> List[complex]:
  Di = D[index]
  C = to_cmplx(D)
  Si = plane_to_sphere(Di)
  coefs = s2pt_to_mobius_coef(Si)
  return [boundaryless_disc_transform(z, coefs) for z in C]


# ---------- spherical cap ----------

def sphere_cap_circle(sp):
  # center slightly inward along normal
  c = sp - (sp / 5.0)
  # correct tangent frame
  xp = np.cross(sp, npole)
  if np.linalg.norm(xp) < MIN_VAL:
    xp = np.array([1.0, 0.0, 0.0])
  u = xp / np.linalg.norm(xp)
  v = np.cross(sp, u)
  uh = fov * u
  vh = fov * v
  t = np.linspace(0, TAU, 360)
  out = []
  for ts in t:
    pt = c + (uh * cos(ts) + vh * sin(ts))
    out.append(pt)
  return np.array(out)


def holonomic_view3(D: Sequence[np.ndarray], index: int):
  sp = plane_to_sphere(D[index])
  trans_pts = to_points(holonomic_rearrangement(D, index))
  scaled_pts = [lambd * p for p in trans_pts]
  lift_pts = np.array([lift_to_sphere_in_direction(p,sp) for p in scaled_pts])
  pers_pt = 2 * sp
  return persp_pt,lift_pts,sphere_cap_circle(sp)



if __name__ == "main":
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  from sphere_curves import random_closed_sphere_curve

























