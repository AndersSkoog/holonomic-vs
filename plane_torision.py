import numpy as np

def torsion_angle(p1, p2):
  a = np.array([-p1[1], p1[0]])
  b = np.array([-p2[1], p2[0]])
  det = a[0]*b[1] - a[1]*b[0]
  return np.arctan2(det,np.dot(a,b))

print(torsion_angle([5,-14.4],[5.5,0.8]))


"""
def frame_delta(dx, dy, v):
    # v = [vx, vy] in local frame
    nabla_x = np.zeros((2,2))  # dx * 0
    nabla_y = np.array([[0, -1],
                        [1, 0]])
    return dx * (nabla_x @ v) + dy * (nabla_y @ v)
"""
