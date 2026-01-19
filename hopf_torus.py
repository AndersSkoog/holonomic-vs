from math import sin, cos
from SU2 import angles
import numpy as np

def toruspoint(c,s,u,v):
    d = (1.0 - s) * sin(v)
    # protect against denom ~ 0 numerically
    if abs(d) < 1e-9: d = 1e-9 if d >= 0 else -1e-9
    x = (c * cos(u)) / d
    y = (c * sin(u)) / d
    z = (s * cos(v)) / d
    return np.array([x,y,z])

def proj_torus(fiber_pt,alpha,us=angles):
  z1,z2 = fiber_pt[0],fiber_pt[1]
  c, s = abs(z1), abs(z2)
  #u = np.asarray(us)
  mer   = toruspoint(c, s, alpha, us)
  par   = toruspoint(c, s, us, alpha)
  vill  = toruspoint(c, s, us, us + alpha)
  vill2 = toruspoint(c, s, us, -us + alpha)
  return mer, par, vill, vill2