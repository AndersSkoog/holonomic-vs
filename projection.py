"""sterographic projection of point in the plane to a point on the sphere centered at [0,0,r]"""
import numpy as np


def plane_to_sphere_vec(pts,r,coef_z):
  points = np.asarray(pts)
  X, Y = points[:,0], points[:,1]
  X2 = X * X
  Y2 = Y * Y
  r2 = r * r
  denom = X2 + Y2 + r2
  x = (2 * r2 * X) / denom
  y = (2 * r2 * Y) / denom
  z = r * (X2 + Y2 - r2) / denom
  ret_a = np.stack((x,y,z),axis=1)
  ret_b = np.stack((x,y,-z),axis=1)
  return np.concatenate([ret_a,ret_b])
  #return np.stack((x, y, z), axis=1), np.stack((x,y,-z),axis=1)


def plane_to_sphere(p, r):
  X, Y = p[0],p[1]
  X2, Y2 = pow(X,2),pow(Y,2)
  r2 = pow(r,2)
  denom = X2 + Y2 + r2
  x = (2 * r2 * X) / denom
  y = (2 * r2 * Y) / denom
  z = r + (r * (X2 + Y2 - r2) / denom)  # center shift up by r
  return [x, y, z]

#  R2 → R3
def stereo_project_R2_R3(p, R=1.0):
  pl = np.asarray(p, dtype=float)
  x, y = pl[..., 0], pl[..., 1]
  r2 = x * x + y * y
  d = r2 + R * R
  X = 2 * R * x / d
  Y = 2 * R * y / d
  Z = (r2 - R * R) / d
  return np.stack([X, Y, Z], axis=-1)