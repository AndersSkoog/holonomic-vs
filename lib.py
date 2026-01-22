import numpy as np
from itertools import combinations
from math import tau, pi

angles = np.linspace(0,tau,360)
t01_100 = np.linspace(0,1,100)

def axis_pairs(dim): return list(combinations(range(dim),2))

def normalize_vector(vec):
  vec = np.array(vec)
  n = np.linalg.norm(vec)
  if n == 0: return vec
  return vec / n

def perpendicular_vector(v, ref=np.array([0, 0, 1])):
  """Return a vector perpendicular to v (cross with reference axis)."""
  v = np.array(v)
  perp = np.cross(v, ref)
  if np.linalg.norm(perp) == 0:
    # v parallel to ref, use another reference
    perp = np.cross(v, np.array([0, 1, 0]))
  return normalize_vector(perp)


def unit(v):
    return v / np.linalg.norm(v)

def orthonormal_frame(v):
    v = unit(v)
    tmp = np.array([1,0,0]) if abs(v[0]) < 0.9 else np.array([0,1,0])
    n1 = unit(np.cross(v, tmp))
    n2 = np.cross(v, n1)
    return n1, n2

def antipodes(p):
  x,y,z = p
  return [
      [x,y,z],[-x,y,z],[x,-y,z],[-x,-y,z],
      [x,y,-z],[-x,y,-z],[x,-y,-z],[-x,-y,-z]
  ]

def orthogonal_ref(plane):
    return [[0,0,1],[0,1,0],[1,0,0]][["xy","xz","yz"].index(plane)]

def orthonormal_u(p,direction):
  ref = orthogonal_ref(direction)
  return normalize_vector(np.cross(ref,p))

def orthonormal_v(p,direction):
  u = orthonormal_u(p,direction)
  return normalize_vector(np.cross(p,u))

