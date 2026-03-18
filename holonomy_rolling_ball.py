


import numpy as np

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v/n


def rot_from_AB(A, B):
    A = normalize(A)
    B = normalize(B)
    axis = np.cross(A, B)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10: return np.eye(3)

    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(A,B), -1.0, 1.0))

    K = np.array([
        [0,-axis[2],axis[1]],
        [axis[2],0,-axis[0]],
        [-axis[1],axis[0],0]
    ])

    O = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
    return O
    
def z_from_orient(O,R):
  pole = np.array([0.0,0.0,1.0]) @ orient
  return R * pole[2]
  

def check_constraint(center, contact,R):
 expected=np.array([contact[0],contact[1],R])
 return np.linalg.norm(center-expected)


def calc_points(config,sc,i,R):
 A=sc[-1] if i ==0 else sc[i-1]
 B=sc[i]
 oA,bA = config
 br = bA[2] # ball radius
 zA = z_from_orient(oA,R) # z frm orient of A
 pA = oA @ A # plane contact of A
 kA = [pA[0],pA[1],zA] # knot point of A
 
 oB = rot_from_AB(A,B) @ oA # orient of B
 zB = z_from_orient(oB,R) # z frm orient of B
 pB = oB @ B # plane contact of B
 dx,dy = pB[0] - pA[0], pB[1] - pA[1]
 bB=[cA[0]+dx,cA[1]+dy,br]#rollballcenter of B
 kB=[pB[0],pB[1],zB] # knot point of B
 return {
   "planeA":pA,
   "knotA":kA,
   "ball_posA":bA,
   "ball_orientA":oA,
   "checkA":check_constraint(bA,pA,R),
   "planeB":pB,
   "knotB":kB,
   "ball_posB":bB,
   "ball_orientB":oB,
   "checkB":check_constraint(bB,pB,R)
 }

  