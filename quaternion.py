import numpy as np
from types import quat
from math import sin,cos,pi,tau,acos,atan2
from constants import MIN_VALUE

def normalize(v): return v / np.linalg.norm(v)

def quat_from_axis_angle(axis, angle) -> quat:
    axis = normalize(axis)
    return np.cos(angle/2), np.sin(angle/2)*axis[0], np.sin(angle/2)*axis[1], np.sin(angle/2)*axis[2]

def quat_mult(q1:quat, q2:quat):
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

