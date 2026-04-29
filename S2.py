import numpy as np
from SO import SO_3, SO_3_UP,SO_3_FWD, SO_3_RIGHT
from lib import angles
from math import pi,tau

xy_circle = np.array([[np.cos(a),np.sin(a),0.0] for a in angles])
yz_circle = np.array([[0.0,np.cos(a),np.sin(a)] for a in angles])
xz_circle = np.array([[np.cos(a),0.0,np.sin(a)] for a in angles])




def antipode(theta,phi):
  atheta = (theta + pi) % tau
  aphi = (phi - pi)
  return atheta,aphi




def stereo_project_R2_R3(p,R=1.0):
  x, y = p
  r2 = x * x + y * y
  d = r2 + R * R
  X = 2 * R * x / d
  Y = 2 * R * y / d
  Z = (r2 - R * R) / d
  return [X,Y,Z]

def R3_to_S2(r3):
  x,y,z = r3
  r = np.sqrt((x*x)+(y*y)+(z*z))
  theta = np.arctan2(y,x)  # longitude
  phi = np.arccos(z / r)  # colatitude
  return np.array([r,theta,phi])


#def R2_to_S2(r2,R): return R3_to_S2(stereo_project_R2_R3(r2,R))

#convert to a spherical coordinate to cartesian coordinate
def S2_to_R3(sc):
  R,theta,phi = sc
  x = R * np.sin(phi) * np.cos(theta)
  y = R * np.sin(phi) * np.sin(theta)
  z = R * np.cos(phi)
  return np.array([x,y,z],dtype=float)

#another conversion suitable for spherical camera
def S2_to_R3_v2(sc):
  r,theta,phi = sc
  x = r * sin(phi) * cos(theta)
  y = r * cos(phi)
  z = r * sin(phi) * sin(theta)
  return np.array([x,y,z],dtype=float)



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
  left = SO_3_RIGHT(so3)
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


def sphere_cap(sp1,sp2):
  # Ensure unit vectors
  p1 = np.asarray(sp1) / np.linalg.norm(sp1)
  p2 = np.asarray(sp2) / np.linalg.norm(sp2)
  #angular distance between p1 and p2
  delta = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
  # Midpoint on the sphere (not normalized yet)
  mid = (p1 + p2) / 2.0
  center = mid / np.linalg.norm(mid)
  # Angular radius of the cap: half the angular distance
  angular_radius = delta / 2.0
  # Area of spherical cap: 2π (1 - cos(angular_radius))
  solid_angle = 2.0 * np.pi * (1.0 - np.cos(angular_radius))






def spherical_cap_between_points(p1, p2):
    """
    Given two points on the unit sphere, compute:
        - the center of the spherical cap that has these two points on its boundary,
        - the angular radius of that cap,
        - the area of the cap (solid angle in steradians).

    The cap is defined as all points whose great-circle distance to the center is ≤ angular_radius.
    The two given points lie on the boundary.

    Parameters:
    p1, p2: array-like of shape (3,), unit vectors.

    Returns:
    dict with keys:
        'center': numpy array (3,), unit vector (midpoint of the arc between p1 and p2)
        'angular_radius': float (radians), half the angular distance between p1 and p2
        'cap_area': float (steradians), area of the spherical cap = 2π (1 - cos(angular_radius))
    """
    # Ensure unit vectors
    p1 = np.asarray(p1) / np.linalg.norm(p1)
    p2 = np.asarray(p2) / np.linalg.norm(p2)

    # Angular distance between p1 and p2
    delta = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))

    # Midpoint on the sphere (not normalized yet)
    mid = (p1 + p2) / 2.0
    center = mid / np.linalg.norm(mid)

    # Angular radius of the cap: half the angular distance
    angular_radius = delta / 2.0

    # Area of spherical cap: 2π (1 - cos(angular_radius))
    cap_area = 2.0 * np.pi * (1.0 - np.cos(angular_radius))

    return {
        'center': center,
        'angular_radius': angular_radius,
        'cap_area': cap_area
    }

# Example
if __name__ == "__main__":
    # Two points on the equator, 90° apart
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])

    result = spherical_cap_between_points(p1, p2)
    print("Center:", result['center'])
    print("Angular radius (rad):", result['angular_radius'])
    print("Angular radius (deg):", np.degrees(result['angular_radius']))
    print("Cap area (sr):", result['cap_area'])

















