import numpy as np
import cmath
from SU2 import S2_to_SU2, SU2_from_r3_sphere_point
from plane_torision import torsion_angle
from projection import plane_to_sphere
from lib import angles

#circle fiber in C2
def base_fiber(ts=angles):
  return np.array([
      [cmath.exp(1j*t), 0]
      for t in ts
  ],dtype=complex)

base_fiber = base_fiber()

def hopf_map(c2):
  z1,z2 = c2
  x=(2*z1.real*z2.real) + (z1.imag * z2.imag)
  y=(2*z1.imag * z2.real) - (z1.real * z2.imag)
  z=pow(abs(z1),2) - pow(abs(z2),2)
  #print(x,y,z)
  return np.array([x,y,z],dtype=float)

def twisted_fiber(U,twist_angle,fiber=base_fiber):
  P = np.array([[np.exp(1j*twist_angle), 0],[0, np.exp(-1j*twist_angle)]],dtype=complex)
  return (U @ P @ fiber.T).T


def hopf_fiber_from_plane_curve_pt(curve_pts,index,R=1):
  p1, p2 = curve_pts[index - 1], curve_pts[index]
  ta = torsion_angle(p1, p2)
  sx,sy,sz = plane_to_sphere(p2,R)
  U = SU2_from_r3_sphere_point(sx,sy,sz)
  fiber = twisted_fiber(U,ta)
  return fiber

def hopf_fiber_from_s2(s2):
  U = S2_to_SU2(s2,1.0)
  return (U @ base_fiber.T).T

def proj_fiber(fiber):return [hopf_map(fp) for fp in fiber]


"""
def torus_circles_from_c2(c2,R=1):
 z1,z2 = c2
 c, s = abs(z1), abs(z2)
 u, v = cmath.phase(z1), cmath.phase(z2)
 d = (1 - s) * np.sin(v)
 print(d)
 if np.isclose(d, 0, 1e-9): d = np.sign(d) * 1e-9
 print(d)
 ts = angles
 alpha = np.zeros(360)
 # meridian torus circle
 mer_circ = [torus_pt(c, s, u, v + t, d) for t in ts]
 par_circ = [torus_pt(c, s, u + t, v, d) for t in ts]
 vil2_circ = [torus_pt(c, s, u, v-t, d) for t in ts]
 vil_circ = [torus_pt(c, s, u, v+t, d) for t in ts]
 return [mer_circ, par_circ, vil_circ, vil2_circ]

def torus_circles_from_plane_pt(p,R=1):
  s2 = plane_to_sphere(p,R)
  z1,z2 = s3_circle_pt(s2)
  c, s = abs(z1), abs(z2)
  u, v = cmath.phase(z1),cmath.phase(z2)
  d = (1 - s) * np.sin(v)
  print(d)
  if np.isclose(d,0,1e-9): d = np.sign(d) * 1e-9
  print(d)
  ts = angles
  alpha = np.zeros(360)
  #meridian torus circle
  mer_circ = [torus_pt(c, s, u, t, d) for t in ts]
  par_circ = [torus_pt(c, s, t, v, d) for t in ts]
  vil2_circ = [torus_pt(c, s, t, -t + alpha,d) for t in ts]
  vil_circ = [torus_pt(c, s, t, t + alpha,d) for t in ts]
  return [mer_circ, par_circ, vil_circ, vil2_circ]

def hopf_fiber_from_s2(s2):
    r,theta,phi = s2
    sr = tan(phi / 2)  # stereographic radius
    a = cmath.rect(sr, theta)  # r * e^{iθ}
    mag = abs(a)
    r = 1 / sqrt(1 + mag**2)
    z1 = r * cmath.exp(1j)
    z2 = a * z1
    return z1,z2

def hopf_map(c2):
  z1,z2 = c2
  x=(2*z1.real*z2.real) + (z1.imag * z2.imag)
  y=(2*z1.imag * z2.real) - (z1.real * z2.imag)
  z=pow(abs(z1),2) - pow(abs(z2),2)
  #print(x,y,z)
  return np.array([x,y,z],dtype=float)


def torus_circles_from_c2(fiber):
    circles = []

    for p in fiber:
        z1, z2 = p
        c, s = abs(z1), abs(z2)
        u, v = cmath.phase(z1), cmath.phase(z2)
        alpha = np.angle(z1) - np.angle(z2)   # ← CRITICAL

        us = angles
        d = (1 - s) * np.sin(v)
        if np.isclose(d, 0, 1e-9): d = np.sign(d) * 1e-9

        circles.append([
            torus_pt(c, s, alpha, u,d),        # meridian
            torus_pt(c, s, u, alpha,d),        # parallel
            torus_pt(c, s, u, u + alpha,d),    # Villarceau
            torus_pt(c, s, u, -u + alpha,d)    # mirrored
        ])

    return circles
"""




