"""
an orientation vector consisting of three angles: $(\theta,\phi,\psi)$
can be used as a convenient and intuitve way to represent rotations and frame-orientations
that can be used in many different transformations and spaces that usually require different inputs,
provided that we know the rules for converting between them.
"""
import SO
import SU2
from types import orient_vec3, unitvec3, vec3, unitquat, quat, orient_axis_ang, zpair, mobius_coef
from SO import Rx, Ry, Rz
from SU2 import SU2
from math import cos, sin
import numpy as np

def vec3_to_unitvec3(v:vec3) -> unitvec3:
  x,y,z = np.asarray(v) / np.linalg.norm(v)
  return x,y,z

def direction_axis(orient_angles:orient_vec3) -> unitvec3:
  theta,phi = orient_angles[0],orient_angles[1] # or is it [0], [2] ??
  x = sin(phi) * cos(theta)
  y = sin(phi) * sin(theta)
  z = np.cos(phi)
  return vec3_to_unitvec3((x,y,z))

def quat_to_unitquat(q:quat) -> unitquat:
  w,i,j,k = q
  n = np.sqrt((w*w)+(i*i)+(j*j)+(k*k))
  return w/n,i/n,j/n,k/n

def axis_ang(orient_angles:orient_vec3) -> orient_axis_ang:
  dir_axis = direction_axis(orient_angles)
  angle = orient_angles[2] # or is it [1] ?
  return dir_axis,angle

def orient_to_quat(orient_angles:orient_vec3) -> unitquat:
  x,y,z = direction_axis(orient_angles)
  angle = orient_angles[2] # or is it [1] ?
  s,w, = sin(angle/2),cos(angle/2)
  i,j,k = x*s,y*s,z*s
  return w,i,j,k

def orient_to_zpair(orient_angles: orient_vec3) -> zpair:
  w,x,y,z = orient_to_quat(orient_angles)
  a = complex(w, x)
  b = complex(y, z)
  return a, b

def orient_to_mobius_coef(orient_angles: orient_vec3) -> mobius_coef:
  a,b = orient_to_zpair(orient_angles)
  c = -b.conjugate()
  d = a.conjugate()
  return a,b,c,d

def orient_to_SO3(orient_angles: orient_vec3):
  theta,phi,psi = orient_angles
  return Rx(theta) @ Ry(phi) @ Rz(psi)

def orient_to_SU2(orient_angles: orient_vec3):
  axis = direction_axis(orient_angles)
  angle = orient_angles[2]
  return SU2(axis,angle)




