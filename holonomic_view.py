
import numpy as np
from constants import MIN_VAL

fov = np.sin(np.pi/5)
gam = np.arccos(4/5)
cap_scale = np.tan(gam/2)
npole = np.array([0.0,0.0,1.0])

def sphere_cap_circle(d,a):
  u = np.cross(d,npole) / np.linalg.norm(np.cross(d,npole))
  v = np.cross(d,u)
  p1 = np.cos(a)*d+np.sin(a)*u
  p2 = np.cos(a)*d+np.sin(a)*v
  p3 = np.cos(a)*d-np.sin(a)*u
  dot1=np.clip(np.dot(p1,p2),-1.0,1.0)
  dot2=np.clip(np.dot(p2,p3),-1.0,1.0)
  ang1=np.arccos(dot1)
  ang2=np.arccos(dot2)
  t = np.linspace(0, 1, 360)


  return



def cap_complex_circle(radius, n=360):
  t = np.linspace(0, 2*np.pi, n)
  return radius * np.exp(1j * t)

def in_unit_disc(p):
 x,y = p[0],p[1]
 return (x*x)+(y*y) <= 1


def to_complex(p):
  x, y = p[0],p[1]
  return complex(x, y)

def complex_disc_curve(plane_curve):
  out = []
  for p in plane_curve:
    z = to_complex(p)
    if in_unit_disc(p): out.append(z)
    else: out.append(1/z)
  return out

def cap_circle(n, alpha=np.pi/5, R=np.tan(np.pi/10), samples=360):
    n = n / np.linalg.norm(n)
    ref = np.array([1.0, 0.0, 0.0])
    dot = abs(np.dot(ref,n))
    if dot > (1 - MIN_VAL):
        ref = np.array([0.0, 1.0, 0.0])

    u = np.cross(n, ref) / np.linalg.norm(np.cross(n,ref))
    v = np.cross(n, u)

    return [
        R * (np.cos(alpha) * n + np.sin(alpha) * (u * np.cos(t) + v * np.sin(t)))
        for t in np.linspace(0, 2*np.pi, samples)
    ]




def inv_stereographic_proj(v: complex):
  m = abs(v)**2
  x = (2*v.real)/(1+m)
  y = (2*v.imag)/(1+m)
  z = (1-m)/(1+m)
  return np.array([x,y,z])

def cone_function(x, d, alpha):
  return np.dot(d, x)**2 - np.dot(x, x) * np.cos(alpha)**2



def holonomic_view_3(contact_curve, sel_index):
  disc_curve = complex_disc_curve(contact_curve)
  curve = contact_curve#complex_disc_curve(contact_curve)
  z_sel = to_complex(curve[sel_index])
  m_sel = abs(z_sel)**2

  a = 1 / np.sqrt(1 + m_sel)
  b = z_sel / np.sqrt(1 + m_sel)
  c = -np.conjugate(z_sel) / np.sqrt(1 + m_sel)
  d = a
  sphere_point = inv_stereographic_proj(z_sel)
  # Möbius-transformed cap
  #z_cap = [((z*a) + b)/((z*c) + d) for z in cap_circle]
  #sphere_cap_circle = [inv_stereographic_proj(z) for z in z_cap]
  #sphere_cap_circle = cap_circle(sphere_point)

  persp_point = sphere_point * 2

  view_points1 = []
  view_points2 = []
  for p in curve:
    z_scaled = cap_scale * to_complex(p)
    z_trans = ((a*z_scaled) + b) / ((c*z_scaled) + d)
    view_points1.append(inv_stereographic_proj(z_trans))

  for z in disc_curve:
    z_scaled = cap_scale * z
    z_trans = ((a*z_scaled) + b) / ((c*z_scaled) + d)
    view_points2.append(inv_stereographic_proj(z_trans))


  return {
    "persp_point": persp_point,
    "disc_curve":disc_curve,
    "view_points1": view_points1,
    "view_points2":view_points2,
    "sphere_point": sphere_point
  }

"""



import numpy as np

fov = np.sin(np.pi/5)
gam = np.arccos(4/5)
lam = np.tan(gam/2)


def cap_circle(n=360):
    t = np.linspace(0, 2*np.pi, n)
    return fov * np.exp(1j * t)


def to_complex(p):
    return complex(p[0], p[1])


def inv_stereographic_proj(v):
    m = abs(v)**2
    return np.array([
        (2*v.real)/(1+m),
        (2*v.imag)/(1+m),
        (1-m)/(1+m)
    ])


def holonomic_view_3(contact_curve, sel_index):

    z_sel = to_complex(contact_curve[sel_index])
    m_sel = abs(z_sel)**2

    a = 1 / np.sqrt(1 + m_sel)
    b = z_sel / np.sqrt(1 + m_sel)
    c = -np.conjugate(z_sel) / np.sqrt(1 + m_sel)
    d = a

    sphere_point = inv_stereographic_proj(z_sel)

    # correct GeoGebra cap: Möbius THEN projection
    z_cap = [(a*z + b)/(c*z + d) for z in cap_circle()]
    sphere_cap = [inv_stereographic_proj(z) for z in z_cap]

    # correct transported curve (NO cap_scale)
    view_points = []
    for p in contact_curve:
        z = to_complex(p)
        zt = (a*(lam*z) + b) / (c*(lam*z) + d)
        view_points.append(inv_stereographic_proj(zt))

    return {
        "sphere_point": sphere_point,
        "view_points": view_points,
        "sphere_cap_circle": sphere_cap
    }

"""



















