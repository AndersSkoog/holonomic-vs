import numpy as np
import pyglet.gl as gl
import pyglet.window as win
import pyglet.graphics as graphics
import pyglet.graphics.shader as shader
import pyglet.gui as gui
import pyglet.app as app
from pyglet_widgets import create_slider
from mesh_lib import sphere_mesh
from utils import mtx_to_glsl_uniform,NxN_id_f32
from math import tau, pi


VERTEX_SRC = """
#version 330 core
layout (location = 0) in vec3 position;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vPos;

void main()
{
    vec4 worldPos = uModel * vec4(position, 1.0);
    vPos = worldPos.xyz;
    gl_Position = uProjection * uView * worldPos;
}
"""

FRAGMENT_SRC = """
#version 330 core
in vec3 vPos;
out vec4 FragColor;

void main()
{
    vec3 n = normalize(vPos);
    float light = dot(n, normalize(vec3(1.0, 1.0, 1.0))) * 0.5 + 0.8;
    FragColor = vec4(vec3(light), 0.4);
}
"""



def perspective(fov, aspect, near, far, **kwargs):
    f = 1.0 / np.tan(fov / 2)
    return np.array([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)

def look_at(cam_pos, cam_target, up_vec, **kwargs):
    f = cam_target - cam_pos
    f /= np.linalg.norm(f)
    r = np.cross(f, up_vec)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)

    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = np.vstack([r, u, -f])
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -cam_pos
    return M @ T

def sphere_look_at(R,cam_theta,cam_phi,**kwargs):
  #theta_rad = np.deg2rad(cam_theta)
  #phi_rad = np.deg2rad(cam_phi)
  x = R * np.sin(cam_phi) * np.cos(cam_theta)
  y = R * np.cos(cam_phi)
  z = R * np.sin(cam_phi) * np.sin(cam_theta)
  cam_pos = np.array([x,y,z],dtype=np.float32)
  return look_at(cam_pos,np.array([0.0,0.0,0.0],dtype=np.float32),np.array([0.0,0.0,1.0],dtype=np.float32))


def createShaderProgram(vert_src,frag_src,uniforms) -> shader.ShaderProgram:
  vert_shader = shader.Shader(vert_src,"vertex")
  frag_shader = shader.Shader(frag_src,"fragment")
  shader_prog = shader.ShaderProgram(vert_shader,frag_shader)
  shader_prog.use()
  for key in uniforms.keys():
    shader_prog[key] = uniforms[key]
  return shader_prog


def uniform_update(pd):
  pd["prog"]["uModel"] = mtx_to_glsl_uniform(NxN_id_f32(N=4))
  pd["prog"]["uView"] = mtx_to_glsl_uniform(sphere_look_at(**pd["params"]))
  pd["prog"]["uProjection"] = mtx_to_glsl_uniform(perspective(**pd["params"]))


mesh = sphere_mesh(radius=1.0,lat_seg=32,lon_seg=32)
mesh_vert,mesh_norm,mesh_ind = mesh[0].flatten(),mesh[1].flatten(),mesh[2].flatten()

batch = graphics.Batch()
prog_win = win.Window(width=800,height=600,caption="sphere test",resizable=False)
prog_params = {
  "R":3.0,
  "cam_theta":40.0,
  "cam_phi":10.0,
  "fov":np.radians(60),
  "aspect":800/600,
  "near":0.1,
  "far":100.0
}
prog_init_uniforms = {
  "uModel":mtx_to_glsl_uniform(NxN_id_f32(N=4)),
  "uView":mtx_to_glsl_uniform(sphere_look_at(**prog_params)),
  "uProjection":mtx_to_glsl_uniform(perspective(**prog_params))
}

prog = createShaderProgram(VERTEX_SRC,FRAGMENT_SRC,prog_init_uniforms)

vlist = prog.vertex_list_indexed(
    count=len(mesh[0]),
    mode=gl.GL_TRIANGLES,
    indices=mesh_ind,
    position=('f',mesh_vert),
    normal=('f',mesh_norm),
    batch=batch
)

#guiwin = win.Window(500,500,"gads")
#guibatch = graphics.Batch()
guiframe = gui.Frame(prog_win,cell_size=16,order=4)


prog_dict = {
  "params":prog_params,
  "guiframe":guiframe,
  "prog":prog,
  "batch":batch
}

slider1,label1 = create_slider(
    param_name="cam_theta",unifupdate=uniform_update,
    wid_index=0,frame_pos=(16,16),cell_size=16,cell_margin=10,
    parser=lambda v: tau * (0.01 * v),prog_dict=prog_dict
)
def slider1_handler(w,val):
    label1.text = f"{val}"
    label1.draw()

slider1.set_handler("on_change",slider1_handler)



@prog_win.event
def on_draw():
  prog_win.clear()
  batch.draw()
  #guibatch.draw()

app.run()




"""
wid_frame = WidgetWindow(
  prog=prog,
  batch=guibatch,
  uniform_update=uniform_update,
  params=prog_params,
  frame_pos=(25, 25)
)

wid_frame.reg_input(
  param_name="cam_theta",
  init_val=(np.pi*2) * 0.4,
  parser=lambda v: (np.pi * 2) * v
)

wid_frame.reg_input(
  param_name="cam_phi",
  init_val=(np.pi*2) * 0.1,
  parser=lambda v: (np.pi * 2) * v
)

wid_frame.reg_input(
  param_name="R",
  init_val=3.0,
  parser=lambda v: 1.0 + (5.0 * v)
)

wid_frame.reg_input(
  param_name="fov",
  init_val=np.radians(60.0),
  parser=lambda v: np.radians(60.0) + (np.radians(30.0) * v)
)
"""