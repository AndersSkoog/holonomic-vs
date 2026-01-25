import numpy as np
from plane_torision import torsion_angle
from hopf import fiber, proj_hopf_link_2
from lib import angles
from SO import SO_3
from S2 import R3_to_S2
from typing import Sequence

def disc_rot(dpts, angle):
  c, s = np.cos(angle), np.sin(angle)
  R = np.array([[c, -s],
                [s,  c]])
  return dpts @ R.T

def disc_to_sphere(dp, R=1.0):
  x, y = dp
  r2 = x*x + y*y
  d = r2 + R*R
  X = 2*R*x / d
  Y = 2*R*y / d
  Z = (r2 - R*R) / d
  theta = np.arctan2(Y, X)
  phi = np.arccos(Z / R)
  return np.array([X,Y,Z]), np.array([theta,phi])

def hopf_link_from_s2(sp,ta,R=1):
  #x1,y1,z1 = sp[0]
  th1,ph1 = sp[1]
  so3 = SO_3(th1,ph1,ta)
  x,y,z = sp[0] @ so3.T
  th2,ph2 = np.arctan2(y, x), np.arccos(z / R)
  fiber1 = fiber(th1,ph1,angles)
  fiber2 = fiber(th2,ph2,angles)
  circ1,circ2 = proj_hopf_link_2(fiber1,fiber2)
  return {"fiber1":fiber1,"fiber2":fiber2,"circ1":circ1,"circ2":circ2}

def holonomic_map(dpts,index:int,R=1.0):
  li = len(dpts) - 1
  assert 0 <= index < li, "index out of range"
  i1 = index
  i2 = 0 if index == li else index + 1
  dp1, dp2 = dpts[i1], dpts[i2]
  ta = torsion_angle(dp1,dp2)
  sp_outer = disc_to_sphere(dp1, R)
  x1,y1,z1 = sp_outer[0]
  theta1,phi1 = sp_outer[1]
  dpts_rot = disc_rot(dpts,theta1)
  sp_inner = disc_to_sphere(dpts_rot[i1],R/2)
  x2,y2,z2 = sp_inner[0]
  theta2,phi2 = sp_inner[1]
  hopf_link = hopf_link_from_s2(sp_inner,ta)
  #hopf_bundle = make a hopf link for each point in dpts_rot



