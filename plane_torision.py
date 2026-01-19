import numpy as np

def torsion_angle(p1, p2):
  a = np.array([-p1[1], p1[0]])
  b = np.array([-p2[1], p2[0]])
  det = a[0]*b[1] - a[1]*b[0]
  return np.arctan2(det,np.dot(a,b))
