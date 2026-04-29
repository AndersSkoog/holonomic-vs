import numpy as np
from constants import MIN_VAL
from lib import periodic_array, non_dup_reverse_array
from math import pi, cos, sin, tau
from S2 import antipode


def normalize(v):
  n = np.linalg.norm(v)
  if n < MIN_VAL: return np.array([1.0, 0.0, 0.0])
  return v / n


def direction(r,theta,phi):
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
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    theta = np.zeros(n)
    phi = np.zeros(n)

    for i in range(1, k+1):               # sum over harmonics
        # random amplitudes and phases for theta and phi separately
        A_theta, B_theta = rng.normal(size=2)
        A_phi, B_phi = rng.normal(size=2)
        phase_theta = rng.uniform(0, 2*np.pi)
        phase_phi = rng.uniform(0, 2*np.pi)

        theta += A_theta * np.cos(i*t + phase_theta) + B_theta * np.sin(i*t + phase_theta)
        phi   += A_phi   * np.cos(i*t + phase_phi)   + B_phi   * np.sin(i*t + phase_phi)


    #atheta = [(v + pi) % tau for v in theta]
    #aphi = [v - pi for v in phi]
    #x = r * np.sin(phi) * np.cos(theta)
    #ax = r * np.sin(aphi) * np.cos(atheta)
    #y = r * np.sin(phi) * np.sin(theta)
    #ay = r * np.sin(aphi) * np.sin(atheta)
    #z = r * np.cos(phi)
    #az = r * np.cos(aphi)
    #out = {
    #  "theta":theta,
    #  "atheta":atheta,
    #  "phi":phi,
    #  "aphi":aphi,
    #  "x":x,
    #  "ax":ax,
    #  "y":y,
    #  "ay":ay,
    #  "z":z,
    #  "az":az
    #}
    #atheta = [antipode(v) for v in theta]
    #aphi = [antipode(v) for v in phi]
    #out_theta = periodic_array(theta)
    #out_phi = periodic_array(phi)
    return theta, phi

def rolltranslation(theta, phi, i, o, p, R):
    n = len(theta)
    prev_i = i-1 if i>0 else -1
    A = direction(R,theta[prev_i], phi[prev_i])
    B = direction(R,theta[i],   phi[i])
    axis_body = normalize(np.cross(A, B))
    angle = np.arccos(np.clip(np.dot(A, B), -1.0, 1.0))
    o_inc = quat_from_axis_angle(axis_body, angle)
    o_new = quat_mult(o, o_inc)
    axis_world = quat_rotate(o, axis_body)[1:]   # rotation axis in world frame
    d = np.array([0.0, 0.0, -1.0])
    move_dir = np.cross(np.asarray(axis_world), d)
    norm_dir = np.linalg.norm(move_dir)
    if norm_dir < MIN_VAL:
        disp = np.zeros(3)
    else:
        move_dir = move_dir / norm_dir
        disp = (R * angle) * move_dir
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
    for i in range(n):
      Di,Oi = rolltranslation(theta,phi,i,last_o,last_p,r)
      pts.append(Di)
      orients.append(Oi)
      last_p = Di
      last_o = Oi
    return pts,orients


  #for j in range(len(CD)):
  #  z = z_from_orient(O[j])
  #  x,y = D[j][0],D[j][1]
  #  M.append(np.array([x,y,z]))


  n,r = 360,0.1
  theta,phi = random_closed_sphere_curve(n=n,seed=36,k=4,r=r) # sphere curve
  #sx = r * np.sin(phi) * np.cos(theta)
  #sy = r * np.sin(phi) * np.sin(theta)
  #sz = r * np.cos(phi)

  pts,orients = construct_disc_curve(theta,phi,r)
  pts2 = [np.array([-p[0],-p[1],0.0]) for p in pts]
  #disc_curve = make_closed(pts)
  xs1,ys1,zs1 = zip(*pts)
  xs2,ys2,zs2 = zip(*pts2)
  fig,ax1 = plt.subplots()
  #ax1 = fig.add_subplot(121, projection='2d')
  ax1.plot(xs1, ys1, 'b-', linewidth=1)
  ax1.plot(xs2, ys2, 'b-', linewidth=1)
  ax1.set_aspect('equal')
  #ax2 = fig.add_subplot(122,projection="3d")
  #ax2.plot(sx,sy,sz,'-b',linewidth=1)

  plt.show()















































  


  
