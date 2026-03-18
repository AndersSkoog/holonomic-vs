from math import cos, sin,acos,pi,tau,radians,sqrt,atan
import constants
import cosmic_date
from typing import List, Tuple, Sequence
import numpy as np
import cmath

from constants import PRECESSION_ANGLE_SEC, SECS_IN_DAY, SPIN_ANGLE_SEC
from cosmic_date import CosmicDate, precession_angle

cmplx_ser = Tuple[float,float] # serialized complex number
unitvec3 = Tuple[float,float,float]
vec3 = Tuple[float,float,float]
quat = Tuple[float,float,float,float]
unitquat = Tuple[float,float,float,float]
zpair = Tuple[complex,complex]
mobius_sphere_orient = Tuple[complex,complex]
mobius_coef = Tuple[complex,complex,complex,complex]
mobius_coef_ser = Tuple[cmplx_ser,cmplx_ser,cmplx_ser,cmplx_ser] #serialized mobius transform coefficents
orient_vec3 = Tuple[float,float,float]
orient_axis_ang = Tuple[unitvec3,float]
minval = 1e-12

def serialize_coef(coef: mobius_coef) -> mobius_coef_ser:
    a, b, c, d = coef
    return (a.real, a.imag), (b.real, b.imag), (c.real, c.imag), (d.real, d.imag)

def deserialize_coef(coef:mobius_coef_ser) -> mobius_coef:
    a,b,c,d = coef
    return complex(a[0],a[1]),complex(b[0],b[1]),complex(c[0],c[1]),complex(d[0],d[1])

"""
Affine Transformations
If z ∈ C, then we can write z = r(cos(θ) + i sin(θ)), where:
r=|z|=sqrt(re(z)²+im(z)²)
θ the argument,arg(z)=arctan(im(z)/re(z))
We have the following five basic maps 
scale(z,v) = z*v, v is real
trasnslate(z,v) = z+v, v is complex
rotate(z,v) = v*z,  v is complex, in other words v =e^iθ=cos(θ) + i sin(θ) 
conj(z) -> re(z) + i -im(z) 
inverte → 1/z
A direct affine transformation is a combination of (1), (2) and (3),
i.e. a map of the form T(z)=Az+B 
"""
def mag(z:complex): return abs(z) # abs(z) = sqrt((z.real*z.real)+(z.imag*z.imag))
def theta(z:complex): return cmath.phase(z) # = math.atan(z.imag/z.real)
def e_i_theta(z:complex): return mag(z)*complex(cos(theta(z)),sin(theta(z)))
def is_unital_to(a:complex,b:complex,unitval:float,tol=minval):
  alpha,beta,gamma,delta = a.real,a.imag,b.real,b.imag
  return np.isclose(pow(alpha,2)+pow(beta,2)+pow(gamma,2)+pow(delta,2),unitval,tol)

def vec3_to_unitvec3(v:vec3) -> unitvec3:
  x,y,z = np.asarray(v) / np.linalg.norm(v)
  return x,y,z

def quat_to_unitquat(q:quat) -> unitquat:
  w,i,j,k = q
  n = np.sqrt((w*w)+(i*i)+(j*j)+(k*k))
  return w/n,i/n,j/n,k/n

def direction_axis(orient_angles:orient_vec3) -> unitvec3:
  theta,phi = orient_angles[0],orient_angles[1] # or is it [0], [2] ??
  x = sin(phi) * cos(theta)
  y = sin(phi) * sin(theta)
  z = np.cos(phi)
  return vec3_to_unitvec3((x,y,z))

def axis_ang(orient_angles:orient_vec3) -> orient_axis_ang:
  dir_axis = direction_axis(orient_angles)
  angle = orient_angles[2] # or is it [1] ?
  return dir_axis,angle

def orient_to_quat(orient_angles:orient_vec3) -> unitquat:
  x,y,z = direction_axis(orient_angles)
  angle = orient_angles[2] # or is it [1] ?
  s,w, = sin(angle/2),cos(angle/2)
  i,j,k = x*s,y*s,z*s
  return w,i,j,k

def orient_to_zpair(orient_angles: orient_vec3) -> zpair:
  w,x,y,z = orient_to_quat(orient_angles)
  a = complex(w, x)
  b = complex(y, z)
  return a, b

def orient_to_mobius_coef(orient_angles: orient_vec3) -> mobius_coef:
  a,b = orient_to_zpair(orient_angles)
  c = -b.conjugate()
  d = a.conjugate()
  return a,b,c,d

"""
orient_vec3 (theta,phi,psi)
theta  → where axis points around z
phi    → how much thet axis tilts
psi    → rotation around that axis

theta – azimuth of the rotation axis (longitude)
phi – polar angle of the axis (colatitude, 0 = north pole)
psi – rotation angle around that axis (the “spin”)
"""

def mobius_trans(z: complex,coef:mobius_coef) -> complex:
    """General Möbius transformation"""
    a,b,c,d = mobius_coef
    denominator = (c * z) + d
    if denominator == 0:
        return float('inf') * 1j  # Return complex infinity
    return ((a * z) + b) / denominator

def mobius_sphere_orient_transform(z:complex,orient:orient_vec3):
 # the matrix: (a b),(c d) forms elements in SU(2)
 return mobius_trans(z,orient_to_mobius_coef(orient))   #((a * z) + b) / ((c * z) + d)

def apply_mobius_trans(points:Sequence[complex],coef:mobius_coef):
  out = []
  for p in points:
    z = mobius_trans(p,coef)
    #r = abs(z)
    #if r > R: z = z / (1 + (r - R))
    #if r > R: z = z / r * R  # clamp to boundary
    out.append(z)
  return out


def earth_rotation_day_animation(points,date:cosmic_date.CosmicDate):
  assert cosmic_date.valid_date(date),"date not valid"
  frame_cnt = constants.SECS_IN_DAY
  start_date = date
  start_date_in_sec = cosmic_date.cosmic_date_in_seconds(date)
  end_date_in_sec = start_date_in_sec + SECS_IN_DAY
  precession_angle_start = PRECESSION_ANGLE_SEC * start_date_in_sec
  precession_angle_end = PRECESSION_ANGLE_SEC * end_date_in_sec
  start_sec = start_date_in_sec % SECS_IN_DAY
  end_sec = (start_sec + SECS_IN_DAY) % SECS_IN_DAY
  spin_angle_start = SPIN_ANGLE_SEC * start_sec
  frames = []
  for i in range(frame_cnt):
    theta = precession_angle_start + (PRECESSION_ANGLE_SEC * i)
    phi = constants.PHI
    psi = (spin_angle_start + (SPIN_ANGLE_SEC * i)) % constants.TAU
    coef = orient_to_mobius_coef((theta,phi,psi))
    frames.append(apply_mobius_trans(points,coef))
  return frames


"""
old code
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from DataFile import DataFile
    from pathlib import Path
    from fourier_curve import angle_and_radial_modulation_curve
    from matplotlib.animation import FuncAnimation
    #from PlotContext import PlotContext
    #from tkiter_widgets import NumberboxInt

    BASE_DIR = Path(__file__).resolve().parent
    precession_coefs_file_path = str(BASE_DIR) + "/data/precession_coefs.json"
    precession_coefs_file = DataFile(precession_coefs_file_path)
    fourier_curve_file_path = str(BASE_DIR) + "/data/fourier_curves.json"
    fourier_curve_file = DataFile(fourier_curve_file_path)
    fourier_curve_params = fourier_curve_file.load("curve_3")
    fourier_curve = angle_and_radial_modulation_curve(**fourier_curve_params)
    fourier_curve_cmplx = [complex(p[0],p[1]) for p in fourier_curve]
    disc_radius = fourier_curve_params["disc_radius"]
    precession_coefs = None
    if precession_coefs_file.has_key("tilt1"):
        precession_coefs_serialized = precession_coefs_file.load("tilt2")["coefs"]
        precession_coefs = [deserialize_coef(coef) for coef in precession_coefs_serialized]
    else:
      precession_tilt = radians(22.5)
      precession_coefs = calc_precession_coefficents(precession_tilt,4)
      precession_coefs_serialized = [serialize_coef(coef) for coef in precession_coefs]
      precession_coefs_file.save("tilt1",{"tilt_angle":22.5,"res":4,"coefs":precession_coefs_serialized})

    frames = calc_frames(fourier_curve_cmplx,precession_coefs,disc_radius)
    frame_cnt = len(frames)
    fig, ax = plt.subplots(figsize=(8, 8))
    disc_radius = 2.0
    padding = 0.2 * disc_radius
    limit = disc_radius + padding
    circle = plt.Circle((0, 0), disc_radius, fill=False, color='black',linewidth=1)
    ax.add_patch(circle)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    line, = ax.plot([], [], 'b-', linewidth=1.5)
    title = ax.set_title('Frame 0')


    def init():
        line.set_data([], [])
        return line, title


    def update(frame_idx):
        frame = frames[frame_idx]
        x = [p.real for p in frame]
        y = [p.imag for p in frame]

        line.set_data(x, y)
        title.set_text(f'Frame {frame_idx + 1}/{len(frames)}')

        return line, title

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        init_func=init,
        interval=50,# (20 fps)
        blit=True,
        repeat=True
    )
    plt.show()
"""