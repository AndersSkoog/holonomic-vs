import numpy as np
from constants import MIN_VAL

def s2_to_r3(theta,phi):
  pr=np.cos(phi)
  x,y,z = pr*np.cos(theta),pr*np.sin(theta),np.sin(phi)
  return np.array([x,y,z])

def z_from_orient(o:(float,float,float,float),R=1.0):
  v = np.array([0.0,0.0,1.0])
  rot_v = quat_rotate(o,v)
  return R * rot_v[3] if R != 1.0 else rot_v[3]


def rolltranslation(sphere_curve,index,contact,R):
  r=1.0
  #index_next = index + 1 if index < len(sphere_curve[0])-1 else 0
  p1 = s2_to_r3(sphere_curve[0][index],sphere_curve[1][index])
  p2 = s2_to_r3(sphere_curve[0][index+1],sphere_curve[1][index+1])
  ang = np.arccos(np.clip(np.dot(p1,p2), -1.0, 1.0))
  rot_axis = np.cross(p1,p2) / np.linalg.norm(np.cross(p1,p2))
  x,y,z = rot_axis
  c,s,C = np.cos(ang),np.sin(ang),1-np.cos(ang)
  R_inc = np.array([
        [c+x*x*C,x*y*C-z*s,x*z*C+y*s],
        [y*x*C+z*s,c+y*y*C,y*z*C-x*s],
        [z*x*C-y*s,z*y*C+x*s,c+z*z*C]
    ])
  R_new = R @ R_inc
  u,v,n = R_new[:, 0],R_new[:, 1],R_new[:, 2]
  d = np.array([0.0, 0.0, -1.0])
  move_dir = np.cross(n, d)
  move_dir_norm = np.linalg.norm(move_dir)
  if move_dir_norm < MIN_VAL:
    disp = np.zeros(3)
  else:
    move_dir = move_dir / move_dir_norm
    disp = (r * ang) * move_dir

  contact_new = contact + disp
  return contact_new,R_new


def inital_R(sphere_curve):
 sc=sphere_curve
 p1,p2 = s2_to_r3(sc[0][0],sc[0][1]),s2_to_r3(sc[1][0],sc[1][1])
 n1,n2 = p1 / np.linalg.norm(p1), p2 / np.linalg.norm(p2)
 t = n2 - n1
 u = t - np.dot(t,n1) * n1
 v = np.cross(n1,u)
 return np.column_stack((u,v,n1))









































  


  
