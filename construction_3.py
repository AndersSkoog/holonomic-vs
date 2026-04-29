from sphere_curves import random_closed_sphere_curve
from holonomic_view import holonomic_view_3
from holonimic_roll import rolltranslation
from orient_vector import orient_to_quat
from quaternion import quat_rotate


def z_from_orient(o,R=1.0):
  v = np.array([0.0,0.0,1.0])
  rot_v = quat_rotate(o,v)
  return R * rot_v[2]


def init_construction():
  S = random_closed_sphere_curve(seed=42) # sphere curve
  D = []
  O = []
  M = []
  p = np.array([0.0,0.0,0.0])
  o = orient_to_quat((0,0,0))
  for i in range(last_ind):
    Di, Oi = rolltranslation(S,i,p,o,1.0)
    D.append(Di)
    O.append(Oi)
    p = Di
    o = Oi
  for j in range(last_ind):
    z = z_from_orient(O[j])
    x,y = D[j]
    M.append(np.array([x,y,zj]))

  return S,D,O,M


S,D,O,M = init_construction()
index = 23
persp_pos,persp_view,sphere_circ = holonomic_view_3(D,index)
























