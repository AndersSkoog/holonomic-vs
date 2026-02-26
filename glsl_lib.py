import numpy as np
from pyglet.math import Mat4, Vec3, Vec4, Vec2
from pyglet.graphics.shader import ShaderProgram
from pyglet.gl import GL_TRIANGLES, GL_POINTS, GL_LINES
from pyglet.graphics import Batch

default_persp_params = {
  "fov":np.radians(60), # 60 degree field of view
  "aspect":800.0/600.0,
  "near":0.1, # 0.1 closest distance in clip space
  "far":100.0 # furthest distance in clip space
}

def transl_mtx(center_pos):
  return Mat4.from_translation(Vec3(*center_pos))

def persp_proj_mtx(aspect:float,near:float,far:float,fov:float):
  return Mat4.perspective_projection(aspect,near,far,fov)

def view_mtx(cam_pos, target_pos):
    return Mat4.look_at(
        Vec3(*cam_pos),
        Vec3(*target_pos),
        Vec3(0.0, 0.0, 1.0)
    )


def mesh_vbo(mesh,prog:ShaderProgram):
  vertices, normals, indices = mesh
  cnt = len(vertices)
  #R, G, B = np.int8(color[0]), np.int8(color[1]), np.int8(color[2])
  #np_col = np.array([R,G,B],dtype=np.int8)
  #vbo_colors = np.full((cnt, 3), np_col).flatten()
  vbo_vertices = np.asarray(vertices,dtype=np.float32).flatten()
  vbo_normals =  np.asarray(vertices,dtype=np.float32).flatten()
  vbo_indices = np.asarray(list(range(cnt)),dtype=np.int32).flatten()
  return prog.vertex_list_indexed(
      count=cnt,
      mode=GL_TRIANGLES,
      indices=vbo_indices,
      position=vbo_vertices,
      normals=vbo_normals,
  )

def points_vbo(points,prog:ShaderProgram):
  cnt = len(points)
  #R, G, B = np.int8(color[0]), np.int8(color[1]), np.int8(color[2])
  #np_col = np.array([R,G,B],dtype=np.int8)
  vbo_points = np.asarray(points,dtype=np.float32).flatten()
  vbo_indices = np.asarray(list(range(cnt)),dtype=np.int32).flatten()
  #vbo_colors = np.full((cnt, 3), np_col).flatten()
  return prog.vertex_list_indexed(
      count=cnt,
      mode=GL_POINTS,
      indices=vbo_indices,
      position=vbo_points,
  )