from fourier_curve import radial_and_angular_modulation_curve
from math import sin,cos,sqrt,pi,tau,atan2
from lib import normalize_vector
from holonomy_rolling_ball import RollTranslation
from contants import MIN_VAL


def plane_to_s2(p):
  x,y = p
  theta = atan2(y,x)
  phi = sqrt(pow(x,2)+pow(y,2))
  #sp = np.array([R*(sin(phi)*cos(theta)),R*(sin(phi)*sin(theta)),R*cos(phi))])
  return (1,theta,phi)


def torsion_angle(knot_pts,i:int):
  l = len(knot_pts)
  i1 = (i-1) % l if (i-1) > 0 else l-1
  i2,i3,i4 = i % l, i+1 % l, i+2 % l
  m1,m2,m3,m4 = knot_points[i1],knot_points[i2],knot_points[i3],knot_points[i4]
  t1,t2,t3 = m2-m1,m3-m2,m4-m3
  b1 = np.cross(t1,t2) / np.linalg.norm(np.cross(t1,t2))
  b2 = np.cross(t2,t3) / np.linalg.norm(np.cross(t2,t3))
  tor_ang = np.arccos(np.dot(b1,b2))
  sign = np.sign(np.dot(np.cross(b1,b2),t2))
  return tor_ang, sign

def base_fiber(res: int):
  angs = np.linspace(0, tau, res, endpoint=False)
  return np.asarray([(complex(sin(a),cos(a)),complex(0.0,0.0)) for a in angs],dtype=complex)

base_fiber_360 = base_fiber(360)


def fiber(R:float,o:quat):
  w,i,j,k = o
  angle,axis = w, np.asarray([i,j,k])
  F = R * base_fiber_360 if R != 1.0 else base_fiber_360
  U = SU2(axis,angle)
  return F @ U


"""
def hopf_link(o:quat,tor:float,dv:vec3,r:float):
 o2 = quat_mult(o,quat_from_axis_angle(dv,tor))
 f1,f2 = fiber(r,o),fiber(r,o2)
 return f1,f2

def hopf_link_2(p,tor):
  px,py = p
  theta,phi = atan2(py,px), sqrt(pow(px,2)+pow(py,2))
  sp1 = np.array([cos(phi)*sin(theta),sin(phi)*sin(theta),cos(phi)])
  o = quat_from_axis_angle(sp1,tor)
  sp2 = quat_rotate(o,sp1)[1:]
  U1 = SU2(sp1,pi)
  U2 = SU2(sp2,pi)
  fiber1 = base_fiber_360 @ U1
  fiber2 = base_fiber_360 @ U2
  return fiber1,fiber2


def hopf_link_persp4_view(dv,tor):
  o = quat_from_axis_angle(dv,tor)
  sp1,sp2 = dv,quat_rotate(o,dv)[:1]
  U1 = SU2(sp1,pi)
  U2 = SU2(sp2,pi)
  fiber1 = base_fiber_360 @ U1
  fiber2 = base_fiber_360 @ U2
  return fiber1,fiber2
"""


def generate_lists_from_rolling_sphere_cruve(curve_pts,roll_sphere_radius:float,embedding_space_radius:float,start_index:int):
  prev_p,prev_o = [0,0],(1,0,0,0) #start pos is origin, start orientation is (1,0,0,0) is this valid?
  orientations = []
  dir_vectors = []
  disc_contact_points = []
  knot_points = []
  torsion_angles = []
  persp3 = []
  d = np.array([0.0,0.0,-1.0])
  pts_cnt = len(curve_pts)
  for i in range(start_index,start_index + pts_cnt):
    prev_i = (i-1) % pts_cnt if i-1 > 0 else pts_cnt - 1
    cur_i = i % pts_cnt
    r1,th1,ph1 = curve_pts[prev_i]
    r2,th2,ph2 = curve_pts[cur_i]
    A = [sin(ph1)*cos(th1),sin(ph1)*sin(th1),cos(ph1)]
    B = [sin(ph2)*cos(th2),sin(ph2)*sin(th2),cos(ph2)]
    axis_body = normalize_vector(np.cross(A,B))
    angle = np.arccos(np.clip(np.dot(A, B), -1.0, 1.0))
    inc_o = quat_from_axis_angle(axis_body,angle)
    new_o = quat_normalize(quat_mult(prev_o,inc_o))
    axis_world = quat_rotate(prev_o,axis_body)[1:]
    move_dir = np.cross(axis_world, d)
    norm_dir = np.linalg.norm(move_dir)
    if norm_dir < MIN_VAL: disp = np.zeros(3)
    else:
        move_dir = move_dir / norm_dir
        # Arc length = R * angle
        disp = (R * angle) * move_dir
    # New contact point
    new_p = prev_p + disp
    knot_z = quat_rotate(new_o,np.array[0.0,0.0,embedding_space_radius])[1:][2]
    knot_points.append(np.array([new_p[0],new_p[1],knot_z]))
    orientations.append(new_o)
    disc_contact_points.append(new_p)
    prev_p = new_p
    prev_o = new_o

  for j in range(start_index,start_index+pts_cnt):
    torsion_angles.append(torsion_angle(knot_points,j))
    r,theta,phi = plane_to_s2(disc_contact_points[j%cnt])
    outer_r = embedding_space_radius * 3
    dir_vec = np.array([sin(phi) * cos(theta),sin(phi) * sin(theta),cos(phi)])
    outer_sp = outer_r * dir_vec
    persp3.append(outer_sp)
    dir_vectors.append(dir_vec)

  R3 = embedding_space_radius*3
  #hopf fibration of 4D perspectives points
  persp4 = [(fiber(R3,orientations[i]),fiber(R3,quat_mult(orientations[i],quat_from_axis_angle(dir_vectors[i],torsion_angles[i])))) for i in range(pts_cnt)]
  #hopf_link(orientations[i],torsion_angles[i],dir_vectors[i],embedding_space_radius*3) for i in range(pts_cnt)]
  geometric_phase = quat_mult(orientations[-1], quat_inverse(orientations[0]))
  ret = {
    "sphere_curve_pts":curve_pts,
    "o":orientations,
    "geometric_phase":geometric_phase,
    "p":disc_contact_points,
    "k":knot_points,
    "tor":torsion_angles,
    "persp3":persp3,
    "persp4":persp4,
    "dir":dir_vectors,
    "cnt":pts_cnt
  }
  return ret

"""
def generate_persp3_view_2(generated_lists,embedding_space_radius:float,index:int):
  roll_orient = generated_lists["o"][index]
  #roll_orient_axis = roll_orient[1:]
  #roll_orient_angle = roll_orient[0]
  #U = SU2(roll_orient_axis,roll_orient_angle)
  sphere_pts = generated_lists["dir"]
  roll_oriented_sphere_pts = [quat_rotate(roll_orient,sp)[1:] for sp in sphere_pts]
  return roll_oriented_sphere_pts
"""

def generate_persp3_view(generated_lists,embedding_space_radius:float,index:int):
  disc_pts = generated_lists["p"]
  x,y = disc_pts[index]
  theta,phi = atan2(y,x), sqrt(pow(x,2)+pow(y,2))
  rm = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
  rot_disc_pts = np.copy(disc_pts) @ rm.T
  view_pts = []
  directions = []
  r = embedding_space_radius
  for p in rot_disc_pts:
    p_x,p_y = p[0],p[1]
    p_theta,p_phi = atan2(p_y,p_x), sqrt(pow(p_x,2)+pow(p_y,2))
    direction = np.array([sin(p_phi)*cos(p_theta),sin(p_phi)*sin(p_theta),cos(p_phi)])
    view_pt = r * direction
    view_pts.append(view_pt)
    directions.append(direction)
  return view_pts, directions

def generate_persp4_view(generated_lists, embedding_space_radius: float, index: int):
    # Get the 3D perspective view (points and directions) for the given index
    pv3, pv3_dirs = generate_persp3_view(generated_lists, embedding_space_radius, index)
    # Rolling orientation of the viewpoint (the same index)
    o_view = generated_lists["o"][index]
    # Rotate all directions by the viewpoint orientation
    pv3_rot = [quat_rotate(o_view, d)[1:] for d in pv3_dirs]
    cnt = len(pv3_rot)
    # Full list of torsion angles (for the original disc points)
    tor_list = generated_lists["tor"]
    hopf_fib = []
    for i in range(cnt):
        dv = pv3_rot[i]          # rotated direction
        tor = tor_list[i]        # torsion for the i‑th original point
        q = quat_from_axis_angle(dv, tor)   # rotation by torsion about dv
        # The two directions for the Hopf link: dv and dv rotated by tor
        sp2 = quat_rotate(q, dv)            # second direction
        U1 = SU2(dv, pi)
        U2 = SU2(sp2,pi)
        fiber1 = base_fiber_360 @ U1
        fiber2 = base_fiber_360 @ U2
        hopf_fib.append((fiber1, fiber2))
    return hopf_fib


"""
def generate_persp4_view(generated_lists,embedding_space_radius:float,index:int):
  p3_pts,p3_dirs = generate_persp3_view(generated_lists,embedding_space_radius,index)
  cnt = generated_lists["cnt"]
  tor_angs = generated_lists["tor"]
  hopf_fib = []
  for i in range(cnt):
    tor,dv = p3_dirs[i],tor_angs[i]
    o = quat_from_axis_angle(dv,tor)
    sp1,sp2 = dv,quat_rotate(o,dv)[1:]
    U1 = SU2(sp1,pi)
    U2 = SU2(sp2,pi)
    fiber1 = base_fiber_360 @ U1
    fiber2 = base_fiber_360 @ U2
    hopf_fib.append((fiber1,fiber2))

  return hopf_fib
"""

def mobius_trans(z: complex,coef:mobius_coef) -> complex:
    """General Möbius transformation"""
    a,b,c,d = mobius_coef
    denominator = (c * z) + d
    if denominator == 0:
        return float('inf') * 1j  # Return complex infinity
    return ((a * z) + b) / denominator

def mobius_sphere_orient_transform(z:complex,orient:orient_vec3):
 # the matrix: (a b),(c d) forms elements in SU(2)
 return mobius_trans(z,orient_to_mobius_coef(orient))   #((a * z) + b) / ((c * z) + d)

def apply_mobius_trans(points:Sequence[complex],coef:mobius_coef):
  out = []
  for p in points:
    z = mobius_trans(p,coef)
    #r = abs(z)
    #if r > R: z = z / (1 + (r - R))
    #if r > R: z = z / r * R  # clamp to boundary
    out.append(z)
  return out


def earth_rotation_day_animation(points,date:cosmic_date.CosmicDate):
  assert cosmic_date.valid_date(date),"date not valid"
  frame_cnt = constants.SECS_IN_DAY
  start_date = date
  start_date_in_sec = cosmic_date.cosmic_date_in_seconds(date)
  end_date_in_sec = start_date_in_sec + SECS_IN_DAY
  precession_angle_start = PRECESSION_ANGLE_SEC * start_date_in_sec
  precession_angle_end = PRECESSION_ANGLE_SEC * end_date_in_sec
  start_sec = start_date_in_sec % SECS_IN_DAY
  end_sec = (start_sec + SECS_IN_DAY) % SECS_IN_DAY
  spin_angle_start = SPIN_ANGLE_SEC * start_sec
  frames = []
  for i in range(frame_cnt):
    theta = precession_angle_start + (PRECESSION_ANGLE_SEC * i)
    phi = constants.PHI
    psi = (spin_angle_start + (SPIN_ANGLE_SEC * i)) % constants.TAU
    coef = orient_to_mobius_coef((theta,phi,psi))
    frames.append(apply_mobius_trans(points,coef))
  return frames





















"""
def generate_persp4_view(generated_lists,embedding_space_radius:float,index:int,view_index:int):
  p3_pts, p3_directions = generate_persp3_view(generated_lists,embedding_space_radius,index)
  q_view = generated_lists["o"][view_index]
  hopf_fibration = []
  cnt = generated_lists["cnt"]
  for i in range(cnt):
    o,tor,dv = generated_lists["o"][i],generated_lists["tor"][i],p3_directions[i]
    q = quat_mult()

    hlink = hopf_link(o,tor,dv,embedding_space_radius)
    hopf_fibration.append(hlink)
  for k in range(generated_lists["cnt"]):
        q_k = generated_lists["o"][k]
        tau_k = generated_lists["tor"][k]
        # Combine with viewpoint orientation
        q_total = quat_mult(q_view, quat_mult(q_k, quat_from_axis_angle(p3_dirs[k], tau_k)))
        fiber = fiber(embedding_space_radius, q_total)  # returns M points in S^3
        circle = stereo_S3_to_R3(fiber)                # project to R^3
        hopf_fibration.append(circle)

    return hopf_fibration
"""
























