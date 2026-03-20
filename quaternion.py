import numpy as np
from types import quat

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
    return quat_mult(q, quat_mult(vq, q_conj))[1:]  # return vector part