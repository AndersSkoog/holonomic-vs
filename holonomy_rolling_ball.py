import numpy as np
from types import quat, unitquat, vec3
from typing import Sequence, Tuple
from orient_vector import direction_axis, orient_to_quat
from quaternion import quat_rotate, quat_from_axis_angle, quat_mult
from lib import normalize_vector

def RollTranslation(L:Sequence[vec3], i:int, o:quat, p:vec3, R:float):
    # Get previous and current spherical coordinates (assuming L stores (r, θ, φ))
    prev_idx = i - 1 if i > 0 else len(L) - 1
    r_prev, theta_prev, phi_prev = L[prev_idx]
    r_cur, theta_cur, phi_cur = L[i]
    # Convert to unit vectors in body coordinates (using direction_axis with ψ=0)
    A = direction_axis((theta_prev, phi_prev, 0.0))
    B = direction_axis((theta_cur, phi_cur, 0.0))
    # Compute incremental rotation that sends A to B
    axis_body = normalize_vector(np.cross(A, B))
    angle = np.arccos(np.clip(np.dot(A, B), -1.0, 1.0))
    # Quaternion for this incremental rotation (body coordinates)
    o_inc = quat_from_axis_angle(axis_body, angle)
    # New orientation
    o_new = quat_mult(o, o_inc)
    # World axis of the incremental rotation
    axis_world = quat_rotate(o, axis_body)
    # Direction of motion on the plane
    d = np.array([0.0, 0.0, -1.0])
    move_dir = np.cross(axis_world, d)
    norm_dir = np.linalg.norm(move_dir)
    if norm_dir < 1e-12: disp = np.zeros(3)
    else:
        move_dir = move_dir / norm_dir
        # Arc length = R * angle
        disp = (R * angle) * move_dir
    # New contact point
    p_new = p + disp
    return p_new, o_new


def z_from_orient(O,R):
  pole = np.array([0.0,0.0,1.0]) @ orient
  return R * pole[2]
  


  