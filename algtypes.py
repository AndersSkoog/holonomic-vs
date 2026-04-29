import numpy as np
#from mytypes import quat
from math import sin,cos,pi,tau,acos,atan2,sqrt
from constants import MIN_VAL

def normalize_vec3(v):
  n = np.linalg.norm(v)
  if n < MIN_VAL: return np.array([1.0, 0.0, 0.0])
  return v / n


class Axis3:

  def __init__(theta:float,phi:float):
    self.vec = normalize_vec3(np.array([sin(phi)*cos(theta),sin(phi)*sin(theta),cos(phi)]))

  def x(self): return self.vec[0]
  def y(self): return self.vec[1]
  def z(self): return self.vec[2]


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


  def rot_vec3(self, vec3):
    a = Quaternion(0.0, vec3[0], vec3[1], vec3[2]) * self.conj()
    b = self * a
    return np.array([b.i,b.j,b.k])

  def to_unit(self):
    w,i,j,k = self.w,self.i,self.j,self.k
    n = sqrt((w*w)+(i*i)+(j*j)+(k*k))
    return Quaternion(w/n,i/n,j/n,k/n)


  @staticmethod
  def from_axis_angle(axis:Axis3,angle:float):
    ha = angle/2
    w,s = cos(ha),sin(ha)
    i,j,k = s * axis.x(), s * axis.y(), s * axis.z()
    return Quaternion(w=w,i=i,j=j,k=k)





