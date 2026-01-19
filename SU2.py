import numpy as np
"""
One construction of elements in SU(2):
An element of SU(2) is a distance traveled from the identity by an amount (a) in a direction specified 
by the cartesian conversion of a point on S² (r,theta,phi).
"""

#pauli matrices
o1 = np.array([[0,1],[1,0]], dtype=complex)
o2 = np.array([[0,-1j],[1j,0]], dtype=complex)
o3 = np.array([[1,0],[0,-1]], dtype=complex)
#Identy in SU(2)
I = np.eye(2, dtype=complex)

#-------------------- Helper Functions---------------------------------
def sphere_to_cart(c):
  r,theta,phi = c
  x = r * np.sin(phi) * np.cos(theta)
  y = r * np.sin(phi) * np.sin(theta)
  z = r * np.cos(phi)
  return np.array([x,y,z])

def normalize_vector(vec):
  vec = np.array(vec)
  n = np.linalg.norm(vec)
  if n == 0: return vec
  return vec / n

def is_SU2(U, tol=1e-9):
  a, b = U.item(0), U.item(1)
  res = pow(abs(a), 2) + pow(abs(b), 2)
  return np.isclose(res,1.0,atol=tol) and np.isclose(np.linalg.det(U), 1.0, atol=tol)

#-------------------------------------------Implementation-------------------------------------------

def SU2(axis,angle):
  nx,ny,nz = axis
  dotsum = nx*o1+ny*o2+nz*o3
  U = np.cos(angle / 2) * I - 1j * np.sin(angle / 2) * dotsum
  assert is_SU2(U), "error"
  return U

def S2_to_SU2(theta,phi,a): return SU2(normalize_vector(sphere_to_cart([1,theta,phi])),a)

def rotation_from_to(sp1,sp2,scalar,tol=1e-9):
    u = sphere_to_cart([1,sp1[1],sp1[2]],"rad")
    v = sphere_to_cart([1,sp2[1],sp2[2]],"rad")
    axis = np.cross(u, v)
    norm = np.linalg.norm(axis)
    if norm < tol:return None,0.0  # same or opposite direction
    axis /= norm
    angle = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
    return SU2(axis,scalar*angle)





