import numpy as np
import cmath
from math import tau, cos, sin
from itertools import combinations

#----------------Utility functions-----------------------------------------------
angles = np.linspace(0, tau, 360)
def axis_pairs(dim): return list(combinations(range(dim), 2))
def normalize_vector(vec):
  vec = np.array(vec)
  n = np.linalg.norm(vec)
  if n == 0: return vec
  return vec / n

#-----------------Disc to S2 Lift----------------------------
def stereo_proj(p, R):
    x, y = p
    r2 = x*x + y*y
    d = r2 + R*R
    X = 2*R*x/d
    Y = 2*R*y/d
    Z = (r2 - R*R)/d
    theta = np.arctan2(Y, X)
    phi   = np.arccos(Z / R)
    return np.array([X, Y, Z]), np.array([theta, phi])

def stereo_proj_vec(P, R=1.0):
    x = P[:,0]
    y = P[:,1]
    r2 = x*x + y*y
    d = r2 + R*R
    X = 2*R*x/d
    Y = 2*R*y/d
    Z = (r2 - R*R)/d
    return np.column_stack([X, Y, Z])

#------------------Rotation matrix SO(N) Group----------------------------------

def axis_rotmtx(i, j, angle, dim):
    mtx = np.identity(dim)
    c, s = np.cos(angle), np.sin(angle)
    mtx[i, i] = c
    mtx[j, j] = c
    mtx[i, j] = -s
    mtx[j, i] = s
    return mtx

def SO(angs):
    N = len(angs)
    rot_mtx = np.identity(N)
    idx_pairs = axis_pairs(N)  # All axis index pairs
    for angle, (i, j) in zip(angles, idx_pairs):
        rot_mtx = axis_rotmtx(i, j, angle, N) @ rot_mtx
    return rot_mtx

#----------------- SU(2) Group -----------------------------------------------

#pauli matrices
o1 = np.array([[0,1],[1,0]], dtype=complex)
o2 = np.array([[0,-1j],[1j,0]], dtype=complex)
o3 = np.array([[1,0],[0,-1]], dtype=complex)
#Identity in SU(2)
I = np.eye(2, dtype=complex)
#base_circle = xy_circle

def is_SU2(U, tol=1e-9):
  a, b = U.item(0), U.item(1)
  res = pow(abs(a), 2) + pow(abs(b), 2)
  return np.isclose(res,1.0,atol=tol) and np.isclose(np.linalg.det(U), 1.0, atol=tol)

def SU2(axis,angle):
  nx,ny,nz = axis
  dotsum = nx*o1+ny*o2+nz*o3
  U = np.cos(angle / 2) * I - 1j * np.sin(angle / 2) * dotsum
  assert is_SU2(U), "error"
  return U

#----------------Holonomic Relation-------------------------

def torision_angle(disc_points,index):
  assert 0 <= index, "index out of range"
  li = len(disc_points) - 1
  if (index + 1) <= li:
    p1,p2 = disc_points[index],disc_points[index+1]
    a,b = [-p1[1],p1[0]], [-p2[1],p2[0]]
    det = a[0] * b[1] - a[1] * b[0]
    return np.arctan2(det, np.dot(a, b))
  else: return 0.0

def rot_disc_points(disc_points, index):
    x, y = disc_points[index]
    theta = np.arctan(y / x)  # polar angle
    rm = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    return np.copy(disc_points) @ rm.T

#-------------------- Hopf fibration -----------------------

def base_fiber(res: int):
  angs = np.linspace(0, tau, res, endpoint=False)
  return np.asarray([(cmath.exp(1j*t),0.0+0.0j) for t in angs],dtype=complex)

base_fiber_360 = base_fiber(360)

def hopf_link_from_disc_point(pts,index,R=1.0):
  assert 0 <= index <= (len(pts)-1), "index out of range"
  tor_ang = torision_angle(pts,index)
  dp = pts[index]
  r3_1,s2 = stereo_proj(dp,R)
  theta,phi = s2
  orient = SO([theta,phi,tor_ang])
  r3_2 = r3_1 @ orient.T  # or should it be orient @ sp[0] ?
  U1 = SU2(r3_1,1.0)
  U2 = SU2(r3_2,1.0)
  fiber1 = base_fiber_360 @ U1
  fiber2 = base_fiber_360 @ U2
  return fiber1,fiber2

def hopf_fibration(pts,R=1.0):
  return [hopf_link_from_disc_point(pts,i,R) for i in range(len(pts))]

def hopf_fiber_to_R4(fiber):
  # fiber shape: (N, 2) complex
  z1 = fiber[:, 0]
  z2 = fiber[:, 1]
  return np.column_stack([z1.real, z1.imag, z2.real, z2.imag])

def stereo_S3_to_R3(w, eps=1e-9):
  denom = 1.0 - w[:,3]
  denom = np.where(np.abs(denom) < eps, eps, denom)
  return w[:, :3] / denom[:, None]

def proj_hopf_fiber(fiber):
  w = hopf_fiber_to_R4(fiber)
  return stereo_S3_to_R3(w)

def proj_hopf_link(link): return proj_hopf_fiber(link[0]),proj_hopf_fiber(link[1])

def proj_hopf_fibration(fibration):
 circles1 = []
 circles2 = []
 for link in fibration:
   fib1,fib2 = link
   circ1,circ2 = proj_hopf_fiber(fib1),proj_hopf_fiber(fib2)
   circles1.append(circ1)
   circles2.append(circ2)
 return circles1,circles2

#-----------------------Holonomic map Implementation----------------------------------
""""
def holonomic_map(disc_points,index,lift):
  assert lift in (0,1,2),"lift higher than 2 not impl"
  disc_pt = disc_points[index]
  rot_disc_pts = rot_disc_points(disc_points,index)
  if lift == 0: return disc_pt,rot_disc_pts
  else:
    sphere_view_pt,mirror_sphere_pts = stereo_proj(disc_pt,3.0), stereo_proj_vec(rot_disc_pts,1.0)
    if lift == 1: return (disc_pt,rot_disc_pts),(sphere_view_pt,mirror_sphere_pts)
    outer_radius_hopf_link = hopf_link_from_disc_point(disc_points,index,3.0)
    inner_radius_hopf_fibration = hopf_fibration(rot_disc_pts,1.0)
    return (disc_pt,rot_disc_pts),(sphere_view_pt,mirror_sphere_pts),(outer_radius_hopf_link,inner_radius_hopf_fibration)
    #return (disc_pt),(sphere_view_pt,mirror_sphere_pts),(outer_radius_hopf_link,inner_radius_hopf_fibration)

#def holomap(disc_points,lift):
#    return [holoview(disc_points,i,lift) for i in range(len(disc_points))]
"""
#-----------------------Visualization----------------------------------------
def sphere_persp_basis(V):
  view_dir = -V / np.linalg.norm(V)
  right = normalize_vector(np.cross(view_dir,np.array([0.0,0.0,1.0])))
  up = normalize_vector(np.cross(right,view_dir))
  return {"view_dir":view_dir,"right":right,"up":up}

def sphere_persp_render_points(P, V, basis, focal=1.0, eps=1e-6):
    # P : (N,3)
    ray = P - V                              # (N,3)
    view_dir = basis["view_dir"]
    right    = basis["right"]
    up       = basis["up"]
    depth = ray @ view_dir                  # (N,)
    depth = np.where(depth < eps, eps, depth)
    scale = focal / depth                   # (N,)
    projected = ray * scale[:, None]        # (N,3)
    x = projected @ right                   # (N,)
    y = projected @ up                      # (N,)
    size = 50.0 / depth
    return np.column_stack([x, y, size])    # (N,3)

def holonomic_view_lift_1(disc_points,index):
  disc_pt = disc_points[index]
  rot_pts = rot_disc_points(disc_points,index)
  view_pt,_ = stereo_proj(disc_pt,3.0)[0]
  inner_sphere_pts = stereo_proj_vec(rot_pts,1.0)
  basis = sphere_persp_basis(view_pt)
  return sphere_persp_render_points(inner_sphere_pts,view_pt,basis,1.0)

def holonomic_view_lift_2(disc_points,index):
  disc_pt = disc_points[index]
  rot_pts = rot_disc_points(disc_points,index)
  view_pt,_ = stereo_proj(disc_pt,3.0)
  basis = sphere_persp_basis(view_pt)
  hopf_fib = hopf_fibration(rot_pts,1.0)
  circles1,circles2 = proj_hopf_fibration(hopf_fib)
  rendpts1 = [sphere_persp_render_points(circle,view_pt,basis,1.0) for circle in circles1]
  rendpts2 = [sphere_persp_render_points(circle,view_pt,basis,1.0) for circle in circles2]
  return rendpts1, rendpts2


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  fig = plt.figure()
  ax = plt.axis('equal')

  def plot_point_list(rend_pts):
    for p in rend_pts:
      x, y, s = p
      plt.plot(x, y, scalex=s, scaley=s)

  def plot_point_lists(rend_pts_arr):
    for arr in rend_pts_arr:
      plot_point_list(arr)

  def draw_0(disc_points,index):
    rend_pts = rot_disc_points(disc_points, index)
    fig.clear()
    for p in rend_pts:
      x,y = p
      plt.plot(x,y,color="black",lw=0.2)

  def draw_1(disc_points,index):
    rend_pts = holonomic_view_lift_1(disc_points, index)
    fig.clear()
    plot_point_list(rend_pts)
    plt.show()

  def draw_2(disc_points,index):
    rend_pts_1,rend_pts_2 = holonomic_view_lift_2(disc_points, index)
    fig.clear()
    plot_point_lists(rend_pts_1)
    plot_point_lists(rend_pts_1)
    plt.show()





















