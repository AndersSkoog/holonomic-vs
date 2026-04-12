import numpy as np
from constants import MIN_VAL
from mobius_trans import apply_mobius_trans
from SO import SO_3
from S2 import S2_to_R3
from typing import Sequence

def to_points(L:Sequence[complex]): -> Sequence[(float,float)] return [(p.real,p.imag) for p in L]
def to_cmplx(L:Sequence[(float,float)]) -> Sequence[complex]: return [complex(p[0],p[1]) for p in L]


def s2pt_to_mobius_coef(s2pt):
    # north pole
    N = np.array([0.0, 0.0, 1.0])
    S = np.asarray(s2pt)
    # axis = cross product
    axis = np.cross(N,S)
    norm = np.linalg.norm(axis)
    if norm < MIN_VAL: return 1+0j, 0+0j, 0+0j, 1+0j
    axis = axis / norm
    dot = np.clip(np.dot(N, S), -1.0, 1.0)
    psi = np.arccos(dot)
    s = np.sin(psi/2)
    w = np.cos(psi/2)
    i, j, k = axis * s
    a = complex(w, i)
    b = complex(j, k)
    c = -np.conj(b)
    d = np.conj(a)
    return a, b, c, d


#assumes mobius coefs correspond to a pure sphere rotation transform
def boundryless_disc_transform(z:complex,coefs:(complex,complex,complex,complex)):
 a,b,c,d = coef
 z1 = apply_mobius_trans(z,coefs)
 if abs(z1) <= 1: return z1 # if z is inside the unit disc just return the mobius trasform of z
 else: return 1 / np.conj(z1) # if z is outside or on disc circle boundry, return the inversion of the mobius transform of z


def inv_stereographic_proj(p:(float,float),R:float):
  x, y = p[0], p[1]
  r2 = x*x + y*y
  d = r2 + R*R
  X = 2*R*x/d
  Y = 2*R*y/d
  Z = (r2 - R*R)/d
  return np.array([X,Y,Z])


#D is a continous curve in the disc, D[index] is an element in that curve.
def holonomic_rearrangement(D,index):
  Di = D[index]
  C = to_cmplx(D) #convert D to complex
  s2pt = inv_stereographic_proj(Di,1) #sphere point associated to Di
  #theta = np.arctan2(sy,sx)  # longitude of sphere point
  #phi = np.sign(y)*np.arccos(sz)  #colatitude of sphere point
  coefs = s2pt_to_mobius_coef(s2pt)
  out = [boundryless_disc_transform(z,coefs) for z in C]
  return out


def holonomic_view3(D, index):
    D_i = D[index]
    C = to_cmplx(D)

    # sphere point
    S_i = inv_stereographic_proj(D_i, 1)
    sx, sy, sz = S_i

    # perspective position
    theta_i = np.arctan2(sy, sx)
    phi_i = np.arccos(sz)
    persp_i = S2_to_R3([3, theta_i, phi_i]) # perspective is loacated on the an outer sphere of radius 3

    # Möbius transform
    coefs = s2pt_to_mobius_coef(S_i)
    C_trans = [boundryless_disc_transform(z, coefs) for z in C]

    # back to sphere
    D_trans = to_points(C_trans)
    D_trans_proj = np.array([inv_stereographic_proj(p,1) for p in D_trans])

    # rotation: align north pole to S_i
    SO = SO_3(theta_i, phi_i, 0)

    D_trans_proj_rot = D_trans_proj @ SO.T

    return {
        "persp_pos": persp_i,
        "persp_view": D_trans_proj_rot
    }


















def tangent_angle(D:Sequence[(float,float)],index:int):
  last_index = len(D) - 1
  adj_prev = D[index-1] if index > 0 else D[-1]
  adj_next = D[index+1] if index < last_index else D[0]
  diff = np.asarray(adj_next) - np.asarray(adj_prev)
  tan_vec = diff / np.linalg.norm(diff)
  x,y = tan_vec[0],tan_vec[1]
  tan_ang = np.arctan2(y,x)
  return tan_ang

def viewpoint_direction(D, index):
  x, y = D[index]
  theta = np.arctan2(y, x)
  psi = tangent_angle(D, index)
  return np.array([np.sin(psi)*np.cos(theta),np.sin(psi)*np.sin(theta),np.cos(psi)])


def holonomic_view3(D:Sequence[(float,float)], index:int):
  theta = np.arctan2(D[index][1],D[index][0])
  phi = tangent_angle(D,index)
  coef = orient_to_mobius_coef(theta,phi,0.0)
  cmplx_pts = to_cmplx(D)
  trans_pts = apply_mobius_trans(cmplx_pts,coef)
  proj_pts = [inv_stereographic_proj((p.real,p.imag),R=1.0) for p in trans_pts]
  return proj_pts

















