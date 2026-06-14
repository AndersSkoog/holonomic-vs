import numpy as np
from constants import MIN_VAL
from lib import periodic_array, non_dup_reverse_array
from math import pi, cos, sin, tau
from S2 import antipode


def normalize(v):
  n = np.linalg.norm(v)
  if n < MIN_VAL: return np.array([1.0, 0.0, 0.0])
  return v / n


def s2_to_r3(r,theta,phi):
  x = r * np.sin(phi) * np.cos(theta)
  y = r * np.sin(phi) * np.sin(theta)
  z = r * np.cos(phi)
  return normalize(np.array([x,y,z]))


def quat_from_axis_angle(axis, angle):
    axis = normalize(axis)
    w,s = np.cos(angle/2),np.sin(angle/2)*axis
    i,j,k = s[0],s[1],s[2]
    return w,i,j,k

def quat_mult(q1, q2):
    #print("q1",q1)
    #print("q2",q2)
    w1,i1,j1,k1 = q1
    w2,i2,j2,k2 = q2
    w3 = (w1*w2) - (i1*i2) - (j1*j2) - (k1*k2)
    i3 = (w1*i2) + (i1*w2) + (j1*k2) - (k1*j2)
    j3 = (w1*j2) - (i1*k2) + (j1*w2) + (k1*i2)
    k3 = (w1*k2) - (i1*j2) + (j1*i2) + (k1*w2)
    return w3,i3,j3,k3

def quat_rotate(q, v):
    # Rotate vector v by quaternion q
    vq = (0, v[0], v[1], v[2])
    q_conj = (q[0], -q[1], -q[2], -q[3])
    return quat_mult(q, quat_mult(vq, q_conj))


def random_closed_sphere_curve(n=360, k=5, r=0.1, seed=None):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2*np.pi, n)
    theta = np.zeros(n)
    phi = np.zeros(n)

    for i in range(1, k+1):
        A_theta, B_theta = rng.normal(size=2)
        A_phi, B_phi = rng.normal(size=2)
        phase_theta = rng.uniform(0, 2*np.pi)
        phase_phi = rng.uniform(0, 2*np.pi)

        theta += A_theta * np.cos(i*t + phase_theta) + B_theta * np.sin(i*t + phase_theta)
        phi   += A_phi   * np.cos(i*t + phase_phi)   + B_phi   * np.sin(i*t + phase_phi)


    return theta, phi

def rolltranslation(r, theta, phi, i, o, p):
    n = len(theta)
    prev_i = i-1 if i>0 else -1
    A = s2_to_r3(r,theta[prev_i], phi[prev_i]) if i > 0 else np.array([0,0,-R])
    B = s2_to_r3(r,theta[i],phi[i])
    axis_body = normalize(np.cross(A, B))
    angle = np.arccos(np.clip(np.dot(A, B), -1.0, 1.0))
    o_inc = quat_from_axis_angle(axis_body, angle)
    o_new = quat_mult(o, o_inc)
    axis_world = quat_rotate(o, axis_body)[1:]
    d = np.array([0.0, 0.0, -1.0])
    move_dir = np.cross(np.asarray(axis_world), d)
    norm_dir = np.linalg.norm(move_dir)
    if norm_dir < MIN_VAL:
        disp = np.zeros(3)
    else:
        move_dir = move_dir / norm_dir
        disp = (r * angle) * move_dir
    p_new = p + disp
    return p_new, o_new


def z_from_orient(o:(float,float,float,float),R=1.0):
  v = np.array([0.0,0.0,1.0])
  rot_v = quat_rotate(o,v)
  return R * rot_v[3] if R != 1.0 else rot_v[3]


def make_closed(pts):
    pts = np.array(pts)
    drift = pts[-1] - pts[0]
    n = len(pts)
    corrected = []
    for i in range(n):
      t = i / (n - 1)
      corrected.append(pts[i] - t * drift)
    return corrected


def boundaryless_disc(z: complex):
  if abs(z) <= 1: return np.array([z.real,z.imag,0.0])
  else:
    z1 = (-1)/np.conj(z)
    return np.array([z1.real,z1.imag,0.0])







if __name__ == "__main__":
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D

  #theta,phi = random_closed_sphere_curve(n=n,seed=16,k=4) # sphere curve
  #S = [np.array([np.sin(phi[i])*np.cos(theta[i]),np.sin(phi[i])*np.sin(theta[i]),np.cos(phi[i])]) for i in range(n)]
  #p = np.array([0.0,0.0,0.0])
  #o = (1.0,0.0,0.0,0.0)

  def construct_disc_curve(theta,phi,r):
    last_o = (1.0,0.0,0.0,0.0)
    last_p = np.array([0.0,0.0,0.0])
    n = len(theta)
    pts = []
    orients = []
    for i in range(1,n):
      p,o = rolltranslation(theta,phi,i,last_o,last_p,r)
      #print(p)
      z = complex(p[0],p[1])
      pts.append(boundaryless_disc(z))
      orients.append(o)
      last_p = p
      last_o = o
    return pts,orients


  #for j in range(len(CD)):
  #  z = z_from_orient(O[j])
  #  x,y = D[j][0],D[j][1]
  #  M.append(np.array([x,y,z]))


  n,r = 360,0.1
  ang = np.linspace(0,tau,n)
  theta,phi = random_closed_sphere_curve(n=n,seed=16,k=4,r=r) # sphere curve
  circle = [np.array([cos(a),sin(a)]) for a in ang]

  #print(theta[0],phi[0])
  #print(theta[-1],phi[0])
  #sx = r * np.sin(phi) * np.cos(theta)
  #sy = r * np.sin(phi) * np.sin(theta)
  #sz = r * np.cos(phi)

  pts,orients = construct_disc_curve(theta,phi,r)
  #pts2 = [np.array([p[0],-p[1],0.0]) for p in non_dup_reverse_array(pts)]
  #pts3 = [np.array([-p[0],p[1],0.0]) for p in non_dup_reverse_array(pts)]
  #pts4 = [np.array([-p[0],-p[1],0.0]) for p in non_dup_reverse_array(pts)]
  #disc_curve = make_closed(pts)
  xs1,ys1,zs1 = zip(*pts)
  cx,cy = zip(*circle)
  #xs2,ys2,zs2 = zip(*pts2)
  #xs3,ys3,zs3 = zip(*pts3)
  #xs4,ys4,zs4 = zip(*pts4)
  fig,ax1 = plt.subplots()
  #ax1 = fig.add_subplot(121, projection='2d')
  ax1.plot(xs1, ys1,'b-',linewidth=0.3)
  ax1.plot(cx,cy,'b-',linewidth=0.3)
  #ax1.plot(xs2, ys2,'b-',linewidth=1)
  #ax1.plot(xs3, ys3,'b-',linewidth=1)
  #ax1.plot(xs4, ys4,'b-',linewidth=1)
  ax1.set_aspect('equal')
  #ax2 = fig.add_subplot(122,projection="3d")
  #ax2.plot(sx,sy,sz,'-b',linewidth=1)
  plt.show()















































  


  
