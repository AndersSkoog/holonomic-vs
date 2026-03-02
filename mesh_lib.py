import numpy as np
from math import tau, pi, sin, cos

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

    return np.array(vertices, dtype=np.float32), np.array(normals, dtype=np.float32), np.array(indices, dtype=np.uint32)


def disc_mesh(radius,res1,res2):
    angles = np.linspace(0,tau,res1)
    radii = np.linspace(0,radius,res2)
    vertices = []
    normals = [[0.0, 0.0, 1.0] for _ in range(res1 * res2)]#costant normals for a flat mesh
    indices = []
    for r in radii:
        for a in angles:
            vertices.append([r*cos(a),r*sin(a),0.0])
    for i in range(res2 - 1):
      for j in range(res1):
          jn = (j + 1) % res1  # wrap angularly
          v00 = i * res1 + j
          v01 = i * res1 + jn
          v10 = (i + 1) * res1 + j
          v11 = (i + 1) * res1 + jn
          # two triangles per quad
          indices.append([v00, v10, v11])
          indices.append([v00, v11, v01])
    return np.array(vertices,dtype=np.float32),np.array(normals,dtype=np.float32),np.array(indices,dtype=np.uint32)




















