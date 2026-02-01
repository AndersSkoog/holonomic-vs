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
    r2 = x * x + y * y
    d = r2 + R * R
    X, Y, Z = (2 * R * x) / d, (2 * R * y) / d, (r2 - R * R) / d
    theta, phi = np.arctan2(Y, X), np.arccos(Z / R)
    return np.array([X, Y, Z]), np.array([theta, phi])
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
  sp = stereo_proj(dp,R)
  theta,phi = sp[1]
  orient = SO([theta,phi,tor_ang])
  p1 = sp[0]
  p2 = p1 @ orient.T  # or should it be orient @ sp[0] ?
  U1 = SU2(p1,1.0)
  U2 = SU2(p2,1.0)
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
def holonomic_map(disc_points,index,lift):
  assert lift in (0,1,2),"lift higher than 2 not impl"
  disc_pt = disc_points[index]
  rot_disc_pts = rot_disc_points(disc_points,index)
  if lift == 0: return disc_pt,rot_disc_pts
  else:
    sphere_view_pt,mirror_sphere_pts = stereo_proj(disc_pt,3.0), [stereo_proj(p,1.0)[0] for p in rot_disc_pts]
    if lift == 1: return (disc_pt,rot_disc_pts),(sphere_view_pt,mirror_sphere_pts)
    outer_radius_hopf_link = hopf_link_from_disc_point(disc_points,index,3.0)
    inner_radius_hopf_fibration = hopf_fibration(rot_disc_pts,1.0)
    return (disc_pt,rot_disc_pts),(sphere_view_pt,mirror_sphere_pts),(outer_radius_hopf_link,inner_radius_hopf_fibration)
    #return (disc_pt),(sphere_view_pt,mirror_sphere_pts),(outer_radius_hopf_link,inner_radius_hopf_fibration)

#def holomap(disc_points,lift):
#    return [holoview(disc_points,i,lift) for i in range(len(disc_points))]

#-----------------------Visualization----------------------------------------
def sphere_perspective_basis(V):
  view_dir = -V / np.linalg.norm(V)
  right = normalize_vector(np.cross(view_dir,np.array([0.0,0.0,1.0])))
  up = normalize_vector(np.cross(right,view_dir))
  return {"view_dir":view_dir,"right":right,"up":up}

def sphere_perspective_render_point_vec(P, V, basis, focal=1.0):
    # P : (N,3)
    # V : (3,)
    ray = P - V                      # (N,3)

    view_dir = basis["view_dir"]     # (3,)
    right    = basis["right"]        # (3,)
    up       = basis["up"]           # (3,)

    # row-wise dot products
    depth = ray @ view_dir           # (N,)

    scale = focal / depth            # (N,)
    projected = ray * scale[:, None] # (N,3)

    x = projected @ right            # (N,)
    y = projected @ up               # (N,)

    return {
        "point": np.column_stack([x, y]),  # (N,2)
        "size": 50.0 / depth               # (N,)
    }
"""
def sphere_perspecitve_render_point(P,V,basis,focal=1.0):
  #basis = sphere_perspective_basis(V)
  ray = P - V
  depth = np.dot(ray, basis["view_dir"])
  scale = focal / depth
  projected = ray * scale
  x = np.dot(projected, basis["right"])
  y = np.dot(projected, basis["up"])
  return {"point":np.asarray([x,y]),"size":50/depth}

def sphere_persp_render_ponints(pts,V,basis,focal=1.0):
  P = pts[:,0]
  ray = P - V
  depth = np.dot(ray, basis["view_dir"])
  scale = focal / depth
  projected = ray * scale
  x = np.dot(projected, basis["right"])
  y = np.dot(projected, basis["up"])
  size = 50.0 / depth
  return np.column_stack([x,y,size])
"""

def holonomic_view_lift_1(disc_points,index):
  data = holonomic_map(disc_points,index,1)
  inner_sphere_pts = data[1][1]
  view_pt = data[1][0]
  basis = sphere_perspective_basis(view_pt)
  return sphere_persp_render_ponints(inner_sphere_pts,view_pt,basis,1.0)

def holonomic_view_lift_2(disc_points,index):
  data = holonomic_map(disc_points,index,2)
  view_pt = data[1][0]
  basis = sphere_perspective_basis(view_pt)
  hopf_fib = data[2][1]
  circles1,circles2 = proj_hopf_fibration(hopf_fib)
  rendpts1 = [sphere_persp_render_ponints(circle,view_pt,basis,1.0) for circle in circles1]
  rendpts2 = [sphere_persp_render_ponints(circle,view_pt,basis,1.0) for circle in circles2]
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
    rend_pts = holonomic_map(disc_points,index,0)[1]
    fig.clear()
    for p in rend_pts:
      x, y, s = p
      plt.plot(x, y, color="black",lw=0.2)

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


"""
#---------------S2 to S3 Lift-------------------------------

def fiber_pt(theta: float, phi: float, t: float) -> (complex, complex):
    e_it = cmath.exp(1j * t)
    z1 = cmath.cos(phi / 2) * e_it
    z2 = cmath.sin(phi / 2) * cmath.exp(1j * (t + theta))
    return z1, z2

def fiber(theta: float, phi: float, tv=angles):
    return np.asarray([fiber_pt(theta, phi, t) for t in tv])

def sphere_point_to_hopf_link(sp, torision, R):
    theta1, phi1 = sp[1]
    orient = SO([theta1, phi1, torision])
    x2, y2, z2 = sp[0] @ orient.T  # or should it be orient @ sp[0] ?
    theta2, phi2 = np.arctan2(y2, x2), np.arccos(z2 / R)
    return fiber(theta1, phi1), fiber(theta2, phi2)

def hopf_fibration(disc_points,R=1.0):
    hopf_links = []
    for i,p in enumerate(disc_points):
        sp = stereo_proj(p,R)
        ta = torision_angle(disc_points,i)
        hl = sphere_point_to_hopf_link(sp, ta, 1.0)
        hopf_links.append(hl)
    return hopf_links
    
def proj_hopf_fiber_2(fiber):
  z1, z2 = fiber[:,0], fiber[:,1]
  denom = 1 - z2.real  # or z2 component along projection axis
  x = z1.real / denom
  y = z1.imag / denom
  z = z2.imag / denom
  return np.asarray([x, y, z])
  
def render_holoview(disc_points,index,lift):
  assert lift in (0,1,2), "lift higher than 2 dimensions not currently implemented"
  data = holoview(disc_points, index, lift)
  if lift == 0: return data[0][0]
  view_point = data[1][0]
  basis = sphere_perspective_basis(view_point)
  if lift == 1:
    mirror_sphere_pts = data[1][1]
    rendered_pts = [sphere_perspecitve_render_point(P,view_point,basis,1.0) for P in mirror_sphere_pts]
    return rendered_pts
  else:
    hopf_fib = data[2][1]
    proj_hopf_fib = [proj_hopf_link(f1,f2) for f1,f2 in hopf_fib]
    rendered_pts = [],[]
    for c1,c2 in proj_hopf_fib:
      for a,b in zip(c1,c2):
        rend_pt_1 = sphere_perspecitve_render_point(a,view_point,basis,1.0)
        rend_pt_2 = sphere_perspecitve_render_point(b,view_point,basis,1.0)
        rendered_pts[0].append(rend_pt_1)
        rendered_pts[1].append(rend_pt_2)
    return rendered_pts
  
  
  
  
"""






















