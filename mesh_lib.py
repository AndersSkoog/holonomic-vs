import numpy as np
import triangle as tr
from math import tau,sin,cos
from lib import to_polar

def adj(l): return [[l[i - 1], l[i]] for i in range(1, len(l))]

def disc_grid_verts(radius, rad_res, ang_res):
    ru = radius / rad_res
    au = tau / ang_res
    verts = [[0.0, 0.0]]
    for ri in range(1, rad_res):
        r = ru * ri
        for ai in range(ang_res):
            a = au * ai
            verts.append([r*cos(a), r*sin(a)])
    return verts

def disc_boundary(radius, ang_res):
    angles = np.linspace(0, tau, ang_res, endpoint=False)
    verts = [[radius*cos(a), radius*sin(a)] for a in angles]
    edges = [[i, (i+1) % ang_res] for i in range(ang_res)]
    return verts, edges

def curve_constraints(curve_pts,offset):
    last_idx = len(curve_pts) - 1
    curve_closed = curve_pts[0] == curve_pts[last_idx]
    verts = [[x, y] for x, y, _ in curve_pts]
    edges = [[offset+(i-1),offset+i] for i in range(1,len(curve_pts))]
    if curve_closed: edges.append([offset+last_idx,offset+1])
    return verts, edges


def disc_mesh_with_curve(radius, rad_res, ang_res, curve_pts):
    ru = radius / rad_res
    # boundary first
    b_verts, b_edges = disc_boundary(radius, ang_res)
    # interior grid
    g_verts = disc_grid_verts(radius-ru,rad_res-1,ang_res)
    verts = b_verts + g_verts
    # curve
    offset = len(verts)
    c_verts, c_edges = curve_constraints(curve_pts, offset)
    verts += c_verts
    edges = b_edges + c_edges
    A = dict(
        vertices=np.array(verts),
        segments=np.array(edges)
    )
    B = tr.triangulate(A, 'pq30')
    verts2d = B['vertices']
    triangles = B['triangles']
    # lift to 3D
    verts3d = [[x, y, 0.0] for x, y in verts2d]
    normals = [[0.0, 0.0, 1.0] for _ in verts3d]
    #curve_verts = verts3d[offset:len(verts3d)]
    return verts3d, triangles, normals, offset


def surface_from_disc_curve(radius, rad_res, ang_res, curve_pts, amplitude=1.0):
    d_verts, d_triangles, d_normals, curve_start_idx = disc_mesh_with_curve(radius, rad_res, ang_res, curve_pts)
    curve_count = len(curve_pts)
    curve_indices = range(curve_start_idx,len(d_verts))
    # Compute cumulative arc length along the curve
    arc_len = [0.0]
    for i in range(1, curve_count):
        p0 = np.array(d_verts[curve_indices[i-1]][:2])
        p1 = np.array(d_verts[curve_indices[i]][:2])
        arc_len.append(arc_len[-1] + np.linalg.norm(p1 - p0))
    arc_len = np.array(arc_len)
    total_len = arc_len[-1]
    t = arc_len / total_len  # normalize to [0,1]
    # Apply sine height function
    for i, vi in enumerate(curve_indices):
        d_verts[vi][2] = amplitude * np.sin(np.pi * t[i])

    return d_verts, d_triangles, d_normals

def sphere_mesh(radius=1.0, lat_seg=32, lon_seg=32):
    vertices = []
    normals = []
    indices = []

    for i in range(lat_seg + 1):
        theta = np.pi * i / lat_seg
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(lon_seg + 1):
            phi = 2 * np.pi * j / lon_seg
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            x = radius * sin_theta * cos_phi
            y = radius * sin_theta * sin_phi
            z = radius * cos_theta

            vertices.append([x, y, z])
            normals.append([x/radius, y/radius, z/radius])

    # indices for triangles
    for i in range(lat_seg):
        for j in range(lon_seg):
            first = i * (lon_seg + 1) + j
            second = first + lon_seg + 1
            indices += [first, second, first + 1, second, second + 1, first + 1]

    return vertices,normals,indices

"""
def disc_mesh(radius,ang_res,rad_res):
    ru = radius / rad_res # concentric radius unit
    au = tau / ang_res # angular unit for calculating evenly spaced vertices along a circle
    vcnt = (ang_res * rad_res) + 1 # vertex count
    rings = [ru*i for i in range(1,rad_res)]
    angles = np.linspace(0,tau,ang_res)
    vertices = [[0.0,0.0,0.0]]
    normals = [[0.0, 0.0, 1.0] for _ in range(vcnt)] #costant normals for a flat mesh
    indices = []
    for r in rings:
        for a in angles:
            vpos = [r*cos(a),r*sin(a),0.0]
            vertices.append(vpos)
    # ---- center fan ----
    for i in range(ang_res):
        j = (i + 1) % ang_res
        indices.append([0, 1 + i, 1 + j])

    # ---- ring connections ----
    for ri in range(1, rad_res - 1):
        ring0 = 1 + (ri - 1) * ang_res
        ring1 = 1 + ri * ang_res

        for j in range(ang_res):
            jn = (j + 1) % ang_res

            v00 = ring0 + j
            v01 = ring0 + jn
            v10 = ring1 + j
            v11 = ring1 + jn

            if (j + ri) % 2 == 0:
                indices.append([v00, v10, v11])
                indices.append([v00, v11, v01])
            else:
                indices.append([v00, v10, v01])
                indices.append([v10, v11, v01])

    return (
        vertices,normals,indices
    )


def find_disc_mesh_quad_for_point(p,radius,rad_res,ang_res):
  r,ang = to_polar(p)
  ru = radius / rad_res
  au = tau / ang_res
  rad_i = min(int(r / ru),rad_res - 2)  # clamp to last valid cell
  ang_i = int(ang / au) % ang_res  # wrap around, dont understand the reason for the modolus here
  vert_idx = lambda ri, ai: 1 + (ang_res * ri) + (ai % ang_res)
  return vert_idx(rad_i,ang_i),vert_idx(rad_i,ang_i+1),vert_idx(rad_i+1,ang_i),vert_idx(rad_i+1,ang_i+1)


def disc_mesh_with_surface_curve(radius, ang_res, rad_res, curve_pts):
    # Get the original disc mesh
    disc_vertices, disc_normals, disc_indices = disc_mesh(radius, ang_res, rad_res)
    last_vert_idx = len(disc_vertices) - 1

    for cp in curve_pts:
        # Create new vertex at xy position of the curve point (z = 0 for now)
        dp = [cp[0], cp[1], 0.0]

        # Find the quad in which this point lies
        vi1, vi2, vi3, vi4 = find_disc_mesh_quad_for_point(dp, radius, rad_res, ang_res)

        # Index of the new vertex
        vi_dp = last_vert_idx + 1

        # Append triangles connecting the new vertex to quad vertices
        disc_indices.append([vi1, vi_dp, vi2])
        disc_indices.append([vi2, vi_dp, vi3])
        disc_indices.append([vi3, vi_dp, vi4])
        disc_indices.append([vi4, vi_dp, vi1])

        # Append new vertex and default normal
        disc_vertices.append(dp)
        disc_normals.append([0.0, 0.0, 1.0])

        # Update last vertex index
        last_vert_idx = vi_dp

    return disc_vertices, disc_normals, disc_indices


def surface_from_disc_curve(radius, rad_res, ang_res, curve_pts, amplitude=1.0):
    d_verts, d_triangles, d_normals, curve_start_idx = \
        disc_mesh_with_curve(radius, rad_res, ang_res, curve_pts)

    curve_count = len(curve_pts)
    curve_indices = range(curve_start_idx, curve_start_idx + curve_count)

    # Compute cumulative arc length along the curve
    arc_len = [0.0]
    for i in range(1, curve_count):
        p0 = np.array(d_verts[curve_indices[i-1]][:2])
        p1 = np.array(d_verts[curve_indices[i]][:2])
        arc_len.append(arc_len[-1] + np.linalg.norm(p1 - p0))
    arc_len = np.array(arc_len)
    total_len = arc_len[-1]
    t = arc_len / total_len  # normalize to [0,1]

    # Apply sine height function
    for i, vi in enumerate(curve_indices):
        d_verts[vi][2] = amplitude * np.sin(np.pi * t[i])

    return d_verts, d_triangles, d_normals
"""




















