import numpy as np
from SO import SO_3, SO_3_UP,SO_3_FWD, SO_3_LEFT
from lib import angles

xy_circle = np.array([[np.cos(a),np.sin(a),0.0] for a in angles])
yz_circle = np.array([[0.0,np.cos(a),np.sin(a)] for a in angles])
xz_circle = np.array([[np.cos(a),0.0,np.sin(a)] for a in angles])

#convert to a spherical coordinate to cartesian coordinate
def S2_to_R3(sc):
  r,theta,phi = sc
  x = r * np.sin(phi) * np.cos(theta)
  y = r * np.sin(phi) * np.sin(theta)
  z = r * np.cos(phi)
  return np.array([x,y,z],dtype=float)

def R3_to_S2(c):
  x,y,z = c
  r = np.sqrt(x * x + y * y + z * z)
  theta = -y*np.arctan2(y,x)
  phi = np.arccos(z/r)
  return np.array([r,theta,phi],dtype=float)

def S2_cube_vertices(sc):
  x,y,z = S2_to_R3(sc)
  return np.array([[x,y,z],[-x,y,z],[x,-y,z],[-x,-y,z],[x,y,-z],[-x,y,-z],[x,-y,-z],[-x,-y,-z]],dtype=float)

def S2_to_SO_3(s2,roll=0):
  r,theta,phi = s2
  return SO_3(theta,phi,roll)

def S2_Orient(s2,roll=0):
  so3 = S2_to_SO_3(s2,roll)
  up = SO_3_UP(so3)
  fwd = SO_3_FWD(so3)
  left = SO_3_LEFT(so3)
  return {"up":up,"fwd":fwd,"left":left,"rot_mtx":so3}

def Oriented_Great_circle(s2,plane,roll=0):
  assert plane in ("xy","yz","xz"), "unexpected plane argument"
  pli = ["xy","yz","xz"].index(plane)
  r = s2[0]
  pts = r * [xy_circle,yz_circle,xz_circle][pli]
  so3 = S2_to_SO_3(s2,roll)
  basis = r * np.eye(3)
  rot_bais = basis @ so3.T
  return pts @ rot_bais.T

def Cortial_circle(s2,roll):
  r,theta,phi = s2
  #sphere paralell circle radius at phi scaled by r
  pr = r * np.sin(phi)
  #sphere meridian circle radius at theta scaled by r
  mr = r * np.cos(theta)
  # base circle in normal xy-plane scaled by pr
  c = pr*xy_circle
  # base circle in normal xz-plane scaled by mr
  pc = mr*xz_circle
  #SO(3) matrix associated to S2 coordinate
  so3 = SO_3(theta,phi,0)
  #oriented circle points
  oc = c @ so3.T
  #orient again with roll angle
  rc = oc @ SO_3(theta,phi,roll).T
  # midpoint moves on perpendicular great circle
  mp = np.array([mr * np.cos(roll), 0, mr * np.sin(roll)])
  return mp + rc

def paralell_circle(s2):
  r,theta,phi = s2
  pr = r * np.sin(phi)
  mr = r * np.cos(theta)
  z = r - (r-mr)
  c = pr*xy_circle + np.full((360,3),[0.0,0.0,z])
  return c

def meridian_circle_yz(s2):
  r,theta,phi = s2
  #pr = r * np.sin(phi)
  mr = r * np.cos(theta)
  x = r * np.sin(phi) * np.cos(theta)
  c = mr*yz_circle + np.full((360,3),[x,0.0,0.0])
  return c

def meridian_circle_xz(s2):
  r,theta,phi = s2
  #pr = r * np.sin(phi)
  mr = r * np.cos(theta)
  y = r * np.sin(phi) * np.sin(theta)
  c = mr*xz_circle + np.full((360,3),[0.0,y,0.0])
  return c
















