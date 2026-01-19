from SU2 import SU2_from_r3_sphere_point, rotate_points, base_fiber, base_circle
from projection import plane_to_sphere
from plane_torision import torsion_angle

def hopf_fiber_from_plane_pt(plane_pt,R=1):
  sx,sy,sz = plane_to_sphere(plane_pt,R)
  U = SU2_from_r3_sphere_point(sx,sy,sz)
  return (U @ base_fiber.T).T

def torus_circle_from_curve_pt(curve_pts,index,R=1):
  p1, p2 = curve_pts[index - 1], curve_pts[index]
  # roll = torsion_angle(p1,p2)
  #theta = np.arccos(sz/R)
  #phi = np.arctan2(sy,sx)

