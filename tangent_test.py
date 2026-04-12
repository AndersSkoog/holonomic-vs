import numpy as np
from quaternion import quat_from_axis_angle

def inv_stereographic_proj(p,R):
  x, y = p
  r2 = x*x + y*y
  d = r2 + R*R
  X = 2*R*x/d
  Y = 2*R*y/d
  Z = (r2 - R*R)/d
  return np.array([X,Y,Z])


def tangent_angle(D,index):
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


def rotation_to_north_pole(v):
  """Quaternion that rotates unit vector v to (0,0,1)."""
  v = v / np.linalg.norm(v)
  w = np.array([0,0,1])
  axis = np.cross(v, w)
  if np.linalg.norm(axis) < 1e-12: return np.array([1,0,0,0]) # identity?
  axis = axis / np.linalg.norm(axis)
  angle = np.arccos(np.dot(v, w))
  return quat_from_axis_angle(axis,angle)


def persp_view3_rotated(D, index, R):
  # 1. Standard lift of all disc points to sphere (southern hemisphere)
  sphere_pts = [inv_stereographic_proj(p, R) for p in D]  # using your function
  # 2. Viewpoint direction
  v = viewpoint_direction(D, index)
  # 3. Rotation that sends v to north pole
  rot = rotation_to_north_pole(v)
  # 4. Apply rotation to all sphere points
  rotated_pts = [rot @ p for p in sphere_pts]
  # 5. Orthographic projection (or stereographic from north pole) to get 2D view
  view_pts = [(p[0], p[1]) for p in rotated_pts]  # orthographic
  # Or if you prefer stereographic from north pole:
  # view_pts = [(p[0]/(1-p[2]), p[1]/(1-p[2])) for p in rotated_pts]
  return view_pts










def persp_pos3(D,index,R):
  x,y = D[index]
  theta = np.arctan2(y,x)
  psi = tangent_angle(D,index)
  r = 3*R
  sx,sy,sz = r*(sin(psi)*cos(theta)),r*(sin(psi)*sin(theta)),r*cos(psi)
  return np.asarray([sx,sy,sz])


def persp_view3(D,index,R):
  x,y = D[index]
  theta = np.arctan2(y,x)
  mat = np.array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])
  rotD = D @ mat
  out = [inv_stereographic_proj(p,R) for p in rotD]
  return out















