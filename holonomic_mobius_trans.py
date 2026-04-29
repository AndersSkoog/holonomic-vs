import numpy as np
from constants import MIN_VAL, TAU, PI
from math import sin, cos, tan, acos, atan2
from mobius_trans import apply_mobius_trans
from SO import SO_3
from S2 import S2_to_R3
from typing import Sequence

fov = sin(PI / 5)
gamma = arccos(4/5)
lambd = tan(gamma/2)
npole = np.array([0.0,0.0,1.0])

def to_points(L:Sequence[complex]): -> Sequence[(float,float)] return [np.array([p.real,p.imag]) for p in L]
def to_cmplx(L) -> Sequence[complex]: return [complex(p[0],p[1]) for p in L]


def plane_to_sphere(p):
 x,y = p
 r = (x*x)+(y*y)
 d = r + 1
 sx,sy,sz = 2 * (x/d), 2 * (y/d), (r-1)/d
 return np.array([sx,sy,-sz])


def view_sphere_circle(D,index):
 p = D[index]
 sp = plane_to_sphere(p)
 c = sp - (sp/5)
 xp = np.cross(sp,npole)
 u = xp / np.linalg.norm(xp)
 v = np.cross(p,u)
 uh,vh = fov * u, fov * v
 uhx,uhy,uhz = uh[0],uh[1],uh[2]
 vhx,vhy,vhz = vh[0],vh[1],vh[2]
 t = np.linspace(0,TAU,360)
 out = [c + np.array([uhx * cos(ts) + vhx * sin(ts),uhy * cos(ts) - vhy * sin(ts)),vhz * sin(ts)] for ts in t]
 return out


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


#D is a continous curve in the disc, D[index] is an element in that curve.
def holonomic_rearrangement(D,index):
  Di = D[index]
  C = to_cmplx(D) #convert D to complex
  Si = plane_to_sphere(D,index) #sphere point associated to Di
  #theta = atan2(,sx)  # longitude of sphere point
  #phi = np.sign(y)*np.arccos(sz)  #colatitude of sphere point
  coefs = s2pt_to_mobius_coef(Si)
  return [boundryless_disc_transform(z,coefs) for z in C]


def holonomic_view3(D,index):
  sp = plane_to_sphere(D,index)
  theta = atan2(sp[1],sp[0])
  phi = acos(sp[2])
  so = SO_3(theta,phi,0.0)
  trans_pts = to_points(holonomic_rearrangement(D,index))
  scaled_pts = [lambd * p for p in trans_pts]
  lift_pts = [plane_to_sphere(p) for p in scaled_pts]
  rot_pts = lift_pts @ so
  return rot_pts




""""
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
"""

















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

















