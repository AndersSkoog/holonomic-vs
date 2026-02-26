import pyglet.app as app
from pyglet.graphics.shader import ShaderProgram, Shader
from pyglet.window import Window
from pyglet.math import Vec3, Vec2, Mat4
from glsl_lib import mesh_vbo, points_vbo, persp_proj_mtx, transl_mtx, default_persp_params
from utils import read_file, proj_file_path
from mesh_lib import sphere_mesh
from DataFile import DataFile
from fourier_curve import angle_and_radial_modulation_curve
from S2 import stereo_project_R2_R3


#pyglet opengl window
win = Window(width=800, height=600, caption="sphere test", resizable=False)

#curve_file_path = proj_file_path("/data/fourier_curves.json")
#curves_file = DataFile(curve_file_path)
#curve_args = curves_file.data["curve_1"]
curve_args = {"disc_radius":1.0,"dc":1.19,"a":[1.2,0.5,0.1,0.3],"b":[1.9,-1.4,0.1,0.7],"phi":[0.4,0.7,42.2,0.7],"psi":[0.2,0.2,5.14,6.73],"M":3,"res":360}
curve_pts = angle_and_radial_modulation_curve(**curve_args)
#print(curve_pts)
sphere = sphere_mesh(1.0)
sel_index = 0
sel_curve_pt = curve_pts[sel_index]
cam_pos = Vec3(*stereo_project_R2_R3(p=sel_curve_pt,R=3.0))
orgin = Vec3(0.0,0.0,0.0)
up = Vec3(0.0,0.0,1.0)

frag_shader_path = proj_file_path("/glsl/light_and_color.frag")
frag_shader_str = read_file(frag_shader_path)
frag_shader = Shader(source_string=frag_shader_str,shader_type='fragment')

mesh_shader_path = proj_file_path("/glsl/mesh_shader.vert")
mesh_shader_str = read_file(mesh_shader_path)
mesh_shader = Shader(source_string=mesh_shader_str,shader_type='vertex')

lift_1_shader_path = proj_file_path("/glsl/holoview_lift_1.vert")
lift1_shader_str = read_file(lift_1_shader_path)
lift1_shader = Shader(source_string=lift1_shader_str,shader_type='vertex')

prog_1 = ShaderProgram(mesh_shader,frag_shader)
prog_1_vbo = mesh_vbo(mesh=sphere,prog=prog_1)
prog_2 = ShaderProgram(lift1_shader,frag_shader)
prog_2_vbo = points_vbo(points=curve_pts,prog=prog_2)

uniforms = {
  "uProjection": persp_proj_mtx(**default_persp_params), # no need to change the default camera parameters
  "uModel": transl_mtx([0.0,0.0,0.0]), # always center world at the orgin
  "uView":Mat4.look_at(cam_pos,orgin,up), # up direction will always be [0,0,1]
  "uCol": Vec3(255, 255, 255), # color for the drawn points is black
  "sel_disc_pont":Vec2(sel_curve_pt[0],sel_curve_pt[1]), # will change according to the selected index of the disc points
  "mesh_col":Vec3(10,30,50), # the color of the mesh will be something light to contrast the black
  "light_add":0.8, #parameter for controlling the light in the fragment shader
  "light_mul":0.5 #--||---
}


@win.event
def on_draw():
  win.clear()
  prog_1.use()
  prog_1_vbo.draw(mode=0)
  prog_2.use()
  prog_2_vbo.draw(mode=0)

app.run()



""""
def index_change(_id,val):
  global sel_index, curve_args, cam_pos, sel_curve_pt
  sel_index = val
  sel_curve_pt = curve_pts[sel_index]
  cam_pos = Vec3(*stereo_project_R2_R3(sel_curve_pt,3.0))
  uniforms["uView"] = Mat4.look_at(cam_pos,orgin,up)
  uniforms["sel_disc_point"] = Vec2(sel_curve_pt[0],sel_curve_pt[1])
  prog_1["uView"] = uniforms["uView"]
  prog_2["uView"] = uniforms["uView"]
  prog_2["sel_disc_point"] = uniforms["sel_disc_point"]


if __name__ == "__main__":
  from PlotContext import PlotContext
  from tkiter_widgets import PresetCtrl, NumberboxInt
  args = {
      "a":[0.0,0.0,0.1,0.0],
      "b":[0.5, 0.0, 0.0, 0.0],
      "phi":[0.0, 0.0, 0.1, 0.0],
      "psi":[0.5, 0.0, 0.0, 0.0],
      "L":2,
      "M":3,
      "res":360
  }
  plt_ctx = PlotContext(-1,1,"cyclic fourier curve",proj="2d")

  def plot_pts():
    plt_ctx.clear()
    plt_ctx.plot_pointlist(curve_pts, "black", 0.3)


  def arg_change(_id,val):
    global curve_pts
    args[_id] = val
    curve_pts = tessellate_fourier_pts(**args)
    plot_pts()


  def load_preset(data):
    print(data)

  index_wid = NumberboxInt(plt_ctx,"index","index",0,len(curve_pts)-1,index_change)
  Preset_ctrl = PresetCtrl(plt_ctx,curve_file_path,lambda:args,load_preset)
  #a_entry = NumberListEntry(plt_ctx,"a","a",args["a"],"float",arg_change)
  #b_entry = NumberListEntry(plt_ctx,"b","b",args["b"],"float",arg_change)
  #phi_entry = NumberListEntry(plt_ctx,"phi","phi",args["phi"],"float",arg_change)
  #psi_entry = NumberListEntry(plt_ctx,"psi","psi",args["psi"],"float",arg_change)
  #L_slider = IntSlider(plt_ctx,"L","L",1,10,3,arg_change)
  #M_slider = IntSlider(plt_ctx,"M","M",1,10,3,arg_change)
  #Preset_ctrl = PresetCtrl(plt_ctx,curve_file_path,lambda:args,load_preset)
  plot_pts()
  plt_ctx.run()
  #win = Window(width=800, height=600, caption="sphere test", resizable=False)
"""
















