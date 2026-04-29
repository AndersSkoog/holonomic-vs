import numpy as np
#from mytypes import quat
from math import sin,cos,pi,tau,acos,atan2,sqrt
from constants import MIN_VAL

def normalize_vec3(v):
  n = np.linalg.norm(v)
  if n < MIN_VAL: return np.array([1.0, 0.0, 0.0])
  return v / n


class Quaternion:

  def __init__(w:float,i:float,j:float,k:float):
    self.w = w
    self.i = i
    self.j = j
    self.k = k


  def __add__(self, q:Quaternion):
    return Quaternion(self.w + q.w, self.i + q.i, self.j + q.j, self.k + q.k)


  def __sub__(self, q:Quaternion):
    return Quaternion(self.w - q.w, self.i - q.i, self.j - q.j, self.k - q.k)


  def __mul__(self,q:Quat):
    w1,i1,j1,k1 = self.w,self.i,self.j,self.k
    w2,i2,j2,k2 = q.w,q.i,q.j,q.k
    w3 = (w1*w2) - (i1*i2) - (j1*j2) - (k1*k2)
    i3 = (w1*i2) + (i1*w2) + (j1*k2) - (k1*j2)
    j3 = (w1*j2) - (i1*k2) + (j1*w2) + (k1*i2)
    k3 = (w1*k2) - (i1*j2) + (j1*i2) + (k1*w2)
    return Quaternion(w=w3,i=i3,j=j3,k=k3)


  def conj(self):
    return Quaternion(self.w, -self.i -self.j, -self.k)


  def norm(self):
    return sqrt(self.w**2 + self.i**2 + self.j**2 + self.k**2)

  @staticmethod
  def from_axis_angle(axis,angle):














def normalize(v): return v / np.linalg.norm(v)

def quat_from_axis_angle(axis, angle) -> quat:
    axis = normalize(axis)
    return np.cos(angle/2), np.sin(angle/2)*axis[0], np.sin(angle/2)*axis[1], np.sin(angle/2)*axis[2]

def quat_mult(q1:quat, q2:quat):
    print(q1)
    print(q2)
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return (w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2)

def quat_rotate(q, v):
    # Rotate vector v by quaternion q
    vq = (0, v[0], v[1], v[2])
    q_conj = (q[0], -q[1], -q[2], -q[3])
    return quat_mult(q, quat_mult(vq, q_conj))  # return vector part


def quat_inverse(q:quat):
  w,i,j,k = q
  return (w,-i,-j,-k)



def quat_components(q:quat):
  w,x,y,z = q
  xx,yy,zz = x*x,y*y,z*z
  psi = 2 * atan2(sqrt(xx+yy+zz),w)
  sin_half = sqrt(xx + yy + zz)
  axis = np.array([x,y,z]) / sin_half if sin_half > MIN_VALUE else np.array([1.0,0.0,0.0])
  phi = atan2(axis[1],axis[0])
  theta = acos(axis[2])
  return {"axis":axis,"theta":theta,"phi":phi,"psi":psi}

