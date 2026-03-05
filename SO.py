import numpy as np
from lib import normalize_vector, axis_pairs,orthonormal_basis
from math import cos, sin


def Rz(yaw):
  c, s = np.cos(yaw), np.sin(yaw)
  return np.array([[c, -s, 0],[s, c, 0],[0, 0, 1]],dtype=float)

def Ry(pitch):
  c, s = np.cos(pitch), np.sin(pitch)
  return np.array([[c, 0, s],[0, 1, 0],[-s, 0, c]],dtype=float)

def Rx(roll):
  c, s = np.cos(roll), np.sin(roll)
  return np.array([[1, 0, 0],[0, c, -s],[0, s, c]],dtype=float)

def Rn(n, i, j, angle):
    R = np.identity(n)
    c, s = np.cos(angle), np.sin(angle)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R

def SO_3(yaw,pitch,roll):return Rz(yaw) @ Ry(pitch) @ Rx(roll)
def SO_3_UP(so3):  return normalize_vector(so3 @ np.array([0.0,0,1.0],dtype=float))
def SO_3_RIGHT(so3):return normalize_vector(so3 @ np.array([0.0,1.0,0.0],dtype=float))
def SO_3_FWD(so3):return normalize_vector(so3 @ np.array([1.0,0.0,0.0],dtype=float))
def SO_N(n,angles):
  assert len(angles) == (n * (n - 1)) // 2, "Incorrect number of angles"
  R = np.identity(n)
  idx_pairs = axis_pairs(n)  # All axis index pairs
  for angle, (i, j) in zip(angles, idx_pairs):
    R = Rn(n, i, j, angle) @ R  # Apply in order
  return R

def rotation_log(R):
    """
    Matrix logarithm for R in SO(3). Returns a skew-symmetric matrix.
    """
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)  # Numerical stability
    theta = np.arccos(cos_theta)
    if np.isclose(theta, 0):
        return np.zeros((3, 3))
    return (theta / (2 * np.sin(theta))) * (R - R.T)

def rotation_exp(S):
    """
    Matrix exponential for a skew-symmetric matrix S (SO(3) -> matrix).
    """
    theta = np.linalg.norm([S[2,1], S[0,2], S[1,0]])
    if np.isclose(theta, 0):
        return np.eye(3)
    A = S / theta
    return np.eye(3) + np.sin(theta) * A + (1 - np.cos(theta)) * (A @ A)

def rotation_angle(R):
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.arccos(cos_theta)

def slerp(R0, R1, t):
    """
    Spherical linear interpolation between R0 and R1 in SO(3)
    t in [0, 1]
    """
    # Relative rotation
    R_rel = R0.T @ R1
    # Logarithm to get rotation vector
    S = rotation_log(R_rel)
    # Scale by t
    S_t = S * t
    # Exponentiate back
    R_t = R0 @ rotation_exp(S_t)
    return R_t

def rotation_axis_angle(R):
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)

    if np.isclose(theta, 0):
        return np.array([0,0,1]), 0.0

    axis = np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ]) / (2*np.sin(theta))

    axis = axis / np.linalg.norm(axis)

    return axis, theta

def torsion_curve_arc(pos_start, frame_start, pos_end, frame_end,
                        t_value=1.0, trace_plane=(1,2),
                        helix_radius=0.1, num_points=40, arc_points=20):
    pos_start = np.asarray(pos_start)
    pos_end = np.asarray(pos_end)
    dir_vec = normalize_vector(pos_end - pos_start)
    # plane perpendicular to the line
    U, V = orthonormal_basis(dir_vec)

    pts = []

    for i in range(num_points):
        t = t_value * i / (num_points-1)
        pos = pos_start + t*(pos_end - pos_start)
        frame_t = slerp(frame_start, frame_end, t)

        # calculate twist angle of chosen axis in plane
        # here we use frame_t rotation in the plane U,V
        # project frame_t axis onto plane
        plane_axis_idx = trace_plane[0]  # pick one axis to track twist
        axis_vec = frame_t[:, plane_axis_idx]
        angle = np.arctan2(np.dot(axis_vec, V), np.dot(axis_vec, U))

        # draw points along the arc from 0 to angle
        for j in range(arc_points):
            a = angle * j / (arc_points-1)
            curve_pt = pos + helix_radius*(cos(a)*U + sin(a)*V)
            pts.append(curve_pt)

    return np.array(pts)


import numpy as np


def torsion_curve(pos_start, frame_start, pos_end, frame_end, t_value, helix_radius=0.1, axis_index=2, num_points=50):
    """
    Compute points along the torsion curve induced by SLERP between two frames.

    Parameters
    ----------
    pos_start : array_like, shape (3,)
        3D start position.
    frame_start : array_like, shape (3,3)
        3x3 rotation matrix at start position (SO(3) frame).
    pos_end : array_like, shape (3,)
        3D end position.
    frame_end : array_like, shape (3,3)
        3x3 rotation matrix at end position (SO(3) frame).
    t_value : float
        Interpolation factor between 0 and 1.
    helix_radius : float
        Distance along chosen axis from the interpolated position.
    axis_index : int
        Which axis of the frame to trace: 0=x,1=y,2=z (default z).
    num_points : int
        Number of points along the curve to generate.

    Returns
    -------
    curve_points : np.ndarray, shape (num_points, 3)
        3D points tracing the torsion curve.
    """
    pos_start = np.asarray(pos_start, dtype=float)
    pos_end = np.asarray(pos_end, dtype=float)
    curve_points = []

    # Interpolation along the segment
    for i in range(num_points):
        t = t_value * (i / (num_points - 1))
        pos_t = pos_start + (pos_end - pos_start) * t
        # SLERP the frame at this fraction
        R_t = slerp(frame_start, frame_end, t)
        # Offset along the chosen frame axis
        offset = R_t[:, axis_index] * helix_radius
        curve_points.append(pos_t + offset)

    return np.array(curve_points)



# Slerp Demo
if __name__ == "__main__":
  from math import tau, sqrt
  from lib import distance
  import matplotlib.pyplot as plt
  from matplotlib.widgets import Slider

  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111, projection='3d')
  plt.subplots_adjust(bottom=0.25)

  def draw_frame(ax_,origin, R, scale=1.0):
      # Columns of R are frame axes: x, y, z
      for vec, color in zip(R.T, ['r', 'g', 'b']):
          ax_.quiver(*origin, *(vec * scale), color=color)

  params = {
    "x1":-1.0,"y1":-1.0,"z1":-1.0,
    "x2": 1.0, "y2": 1.0, "z2": 1.0,
    "Rx1":tau*0.0,"Ry1":tau*0.1,"Rz1":tau*0.0,
    "Rx2":tau*0.3,"Ry2":tau*0.7,"Rz2":tau*0.4,
    "t":0.0,"tor_axis":0
  }

  points_labels = ["start_pos", "end_pos", "t_pos"]

  def update():
      ax.clear()
      # Get positions
      pos1 = np.array([params['x1'], params['y1'], params['z1']])
      pos2 = np.array([params['x2'], params['y2'], params['z2']])
      vec = pos2 - pos1
      t = params['t']
      pos_t = pos1 + vec * t

      # Get frames
      R1 = SO_3(params['Rx1'], params['Ry1'], params['Rz1'])
      R2 = SO_3(params['Rx2'], params['Ry2'], params['Rz2'])
      R_t = slerp(R1, R2, t)
      tor_res = 50
      tor_plane = [(0,1),(0,2),(1,2)][params["tor_axis"]]
      tor_col =  ["red","green","blue"][params["tor_axis"]]
      #tca_x,tca_y,tca_z = zip(*torsion_curve_arc(pos1,R1,pos2,R2,t,tor_plane,0.3,tor_res,tor_res))
      tc0_x,tc0_y,tc0_z = zip(*torsion_curve(pos1,R1,pos2,R2,t,0.15,0,tor_res))
      tc1_x,tc1_y,tc1_z = zip(*torsion_curve(pos1,R1,pos2,R2,t,0.15,1,tor_res))
      tc2_x,tc2_y,tc2_z = zip(*torsion_curve(pos1,R1,pos2,R2,t,0.15,2,tor_res))


      #tc2_x, tc2_y, tc2_z = zip(*torsion_curve_arc(pos1, R1, pos2, R2, t, (0,2), 0.3, tc_res, tc_res))
      #tc3_x, tc3_y, tc3_z = zip(*torsion_curve_arc(pos1, R1, pos2, R2, t, (1,2), 0.3, tc_res, tc_res))
      ax.text(params["x1"] + 0.3,params["y1"],params["z1"],"start_pos")
      ax.text(params["x2"] + 0.3,params["y2"],params["z2"],"end_pos")
      ax.text(pos_t[0] + 0.3,pos_t[1],pos_t[2],"t_pos")
      #ax.plot(xs=tca_x,ys=tca_y,zs=tca_z,lw=0.3,color=tor_col)
      ax.plot(xs=tc0_x,ys=tc0_y,zs=tc0_z,lw=1.0,color="red")
      ax.plot(xs=tc1_x,ys=tc1_y,zs=tc1_z,lw=1.0,color="green")
      ax.plot(xs=tc2_x,ys=tc2_y,zs=tc2_z,lw=1.0,color="blue")

      ax.plot(xs=[pos1[0],pos_t[0]],ys=[pos1[1],pos_t[2]],zs=[pos1[2],pos_t[2]],lw=1.0,color="purple")
      #ax.plot(xs=tc2_x,ys=tc2_y,zs=tc2_z,lw=0.3,color="green")
      #ax.plot(xs=tc3_x,ys=tc3_y,zs=tc3_z,lw=0.3,color="blue")






      # Plot frames
      draw_frame(ax, pos1, R1)
      draw_frame(ax, pos2, R2)
      draw_frame(ax, pos_t, R_t)

      ax.set_xlim(-1, 1)
      ax.set_ylim(-1, 1)
      ax.set_zlim(-1, 1)
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
      fig.canvas.draw_idle()


  # Initial plot
  update()

  # -------------------
  # Sliders
  # -------------------
  ui_el_width = 0.1
  ui_el_height = 0.03
  ui_el_gap = 0.01
  ui_left = 0.03
  ui_bottom = 0.0

  def calc_ui_ax(elnumber:int):
    ax_y = ui_bottom + ((ui_el_gap + ui_el_height) * elnumber) if elnumber > 0 else ui_bottom
    return plt.axes((ui_left,ax_y,ui_el_width,ui_el_height))

  def slider_change(key,val):
      params[key] = val
      update()

  def create_slider(num,key,min_val,max_val,step_val,init_val,color):
    ret = Slider(ax=calc_ui_ax(num),label=key,valinit=init_val,valmin=min_val,valmax=max_val,valstep=step_val,track_color=color)
    ret.on_changed(lambda v: slider_change(key,v))
    return ret


  t_slider = create_slider(1,"t", 0.0, 1.0, 0.01,params["t"],"black")
  tor_axis_sel = create_slider(2, "tor_axis", 0, 2, 1, params["tor_axis"], "black")
  x1_slider = create_slider(3,"x1",-1.0,1.0,0.01,params["x1"],"red")
  y1_slider = create_slider(4,"y1",-1.0,1.0,0.01,params["y1"],"red")
  z1_slider = create_slider(5,"z1",-1.0,1.0,0.01,params["z1"],"red")
  Rx1_slider = create_slider(6,"Rx1",0.0,tau,0.01,params["Rx1"],"red")
  Ry1_slider = create_slider(7,"Ry1",0.0,tau,0.01,params["Ry1"],"red")
  Rz1_slider = create_slider(8,"Rz1",0.0,tau,0.01,params["Rz1"],"red")
  x2_slider = create_slider(9,"x2",-1.0,1.0,0.01,params["x2"],"blue")
  y2_slider = create_slider(10,"y2",-1.0,1.0,0.01,params["y2"],"blue")
  z2_slider = create_slider(11,"z2",-1.0,1.0,0.01,params["z2"],"blue")
  Rx2_slider = create_slider(12,"Rx2",0.0,tau,0.01,params["Rx2"],"blue")
  Ry2_slider = create_slider(13,"Ry2",0.0,tau,0.01,params["Ry2"],"blue")
  Rz2_slider = create_slider(14,"Rz2",0.0,tau,0.01,params["Rz2"],"blue")


  plt.show()







