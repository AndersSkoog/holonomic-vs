import numpy as np
from itertools import combinations
from math import tau, pi

angles = np.linspace(0,tau,360)
t01_100 = np.linspace(0,1,100)

def axis_pairs(dim): return list(combinations(range(dim),2))

def normalize_vector(vec):
  vec = np.asarray(vec)
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

def NxN_id_f16(N:int):return np.eye(N,dtype=np.float16)
def NxN_id_f32(N:int):return np.eye(N,dtype=np.float32)
def NxN_id_i16(N:int):return np.eye(N,dtype=np.int16)
def NxN_id_i32(N:int):return np.eye(N,dtype=np.int32)
def arr_f32(vals): return np.array(vals,dtype=np.float32)
def arr_f16(vals): return np.array(vals,dtype=np.float16)
def arr_i32(vals): return np.array(vals,dtype=np.int32)
def arr_i16(vals): return np.array(vals,dtype=np.int16)
def arr_fill_f32(val:float,size:int): np.full(shape=(size,),fill_value=val,dtype=np.float32)
def arr_fill_f16(val: float, size: int): np.full(shape=(size,), fill_value=val, dtype=np.float16)
def arr_fill_i32(val: float, size: int): np.full(shape=(size,), fill_value=val, dtype=np.float32)
def arr_fill_i16(val: float, size: int): np.full(shape=(size,), fill_value=val, dtype=np.float16)
def mtx_to_glsl_uniform(mtx:np.ndarray): return mtx.T.flatten()
def arr_to_glsl_uniform(arr:np.array): return arr.T.flatten()