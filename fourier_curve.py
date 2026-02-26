from math import tau, cos, sin
import numpy as np
import cmath
from DataFile import DataFile
from utils import proj_file_path

def fourier_series_trunc(dc, a, b, k, t):
    """
    Compute a truncated Fourier series up to harmonic k
    dc: float
        The zero-frequency (constant) term.
    a, b: lists or arrays
        Cosine and sine coefficients for all harmonics in the series. Length must be >= num_harmonics.
    k: int
        Number of harmonics to include (1 = fundamental, etc.)
    t: float
        Parameter (can be any real number).
        because the beauty of the fourier series comes from its unit periodicity that will wrap around 1,
        regardless of the value of t
    """
    assert len(a) == len(b), "Coefficient arrays not of same length"
    assert len(a) >= k and len(b) >= k, "Coefficient arrays too short"
    #harmonics = [n for n in range(1,k+1)]
    result = dc + sum([(a[h-1] * cos(tau * h * t)) + (b[h-1] * sin(tau * h * t)) for h in range(1,k+1)])
    return result

def fourier_series(dc,a,b,t):return fourier_series_trunc(dc,a,b,len(a),t)

def scale_coef_to_radius(dc,a,b,R):
  coef_sum = abs(dc) + sum(abs(x) + abs(y) for x, y in zip(a, b))
  scale = R / coef_sum
  dc_scaled = dc * scale
  a_scaled = [x * scale for x in a]
  b_scaled = [y * scale for y in b]
  return dc_scaled,a_scaled,b_scaled

"""
Planar curves functions based on fourier series
"""
def radial_modulation_curve(disc_radius,dc,a,b,res,**kwargs):
  t_vals = np.linspace(0, 1, res)
  dc_scaled,a_scaled,b_scaled = scale_coef_to_radius(dc,a,b,disc_radius)
  r_vals = np.array([fourier_series(dc_scaled, a_scaled, b_scaled, t) for t in t_vals])
  return [[r * cos(tau * t),r * sin(tau*t)] for r,t in zip(r_vals,t_vals)]

def angle_modulation_curve(disc_radius,dc,phi,psi,M,res,**kwargs):
  t_vals = np.linspace(0, 1, res)
  dc_scaled,phi_scaled,psi_scaled = scale_coef_to_radius(dc,phi,psi,disc_radius)
  theta_vals = tau * M * t_vals + np.array([fourier_series(0, phi_scaled, psi_scaled, t) for t in t_vals])
  return [[dc_scaled * cos(theta), dc_scaled * sin(theta)] for theta in theta_vals]

def angle_and_radial_modulation_curve(disc_radius,dc,a,phi,b,psi,M,res,**kwargs):
    t_vals = np.linspace(0, 1, res)
    dc_scaled,a_scaled,b_scaled = scale_coef_to_radius(dc,a,b,disc_radius)
    dummy,phi_scaled,psi_scaled = scale_coef_to_radius(dc,phi,psi,disc_radius)
    r_vals = np.array([fourier_series(dc_scaled,a_scaled,b_scaled,t) for t in t_vals])
    theta_vals = tau * M * t_vals + np.array([fourier_series(0, phi_scaled, psi_scaled, t) for t in t_vals])
    return [[r * cos(theta),r * sin(theta)] for r,theta in zip(r_vals,theta_vals)]

if __name__ == "__main__":
  from PlotContext import PlotContext
  from tkiter_widgets import FloatSlider,IntSlider, NumberListEntry, SelectBox
  pctx = PlotContext(-1, 1, "fourier curves", proj="2d")
  args = {"disc_radius":1.0,"dc":0.6,"a":[0.2,0.5,0.7,0.3],"b":[0.9,0.4,0.1,0.7],"phi":[0.4,0.7,0.2,0.7],"psi":[0.7,0.2,0.34,0.73],"M":1,"res":360}
  curve_func_key = "radial_modulation"
  func_dict = {
      "radial_modulation":radial_modulation_curve,
      "angle_modulation":angle_modulation_curve,
      "angle_and_radial_modulation":angle_and_radial_modulation_curve
  }
  pts = func_dict[curve_func_key](**args)
  circ = [[args["disc_radius"]*cos(a),args["disc_radius"]*sin(a)] for a in np.linspace(0,tau,360)]

  def plot_pts():
   pctx.clear()
   pctx.plot_pointlist(pts,"black",0.3)
   pctx.plot_pointlist(circ,"purple",0.3)

  def arg_change(_id,val):
    global pts, curve_func_key, circ
    if _id == "curve_func": curve_func_key = val
    else:
      args[_id] = val
      if _id == "disc_radius":
        circ = [[args["disc_radius"]*cos(a),args["disc_radius"]*sin(a)] for a in np.linspace(0,tau,360)]
    pts = func_dict[curve_func_key](**args)
    plot_pts()

  fn_sel = SelectBox(pctx,"curve_func","curve_func",list(func_dict.keys()),arg_change)
  radius_wid = FloatSlider(pctx,"disc_radius","disc_radius",0.01,5.0,1.0,arg_change)
  dc_wid = FloatSlider(pctx,"dc","dc",0.01,5.0,0.6,arg_change)
  M_wid = IntSlider(pctx,"M","M",1,10,1,arg_change)
  a_wid = NumberListEntry(pctx,"a","a",args["a"],"float",arg_change)
  b_wid = NumberListEntry(pctx,"a","b",args["b"],"float",arg_change)
  phi_wid = NumberListEntry(pctx,"phi","phi",args["phi"],"float",arg_change)
  psi_wid = NumberListEntry(pctx,"psi","psi",args["psi"],"float",arg_change)


  pctx.run()



#print(radial_modulation_curve(0.6,[0.2,0.7,0.3,0.9],[0.53,0.23,0.84,0.64],360))

#dc_val = 0.4
#coef_a = [0.6,0.2,0.7,0.82]
#coef_b = [0.23,0.34,0.12,0.73]
#t_vals = np.linspace(0,1,100)
#fourier_series_expansion_list = [fourier_series_expansion(dc_val,coef_a,coef_b,t) for t in t_vals]
#coef_a = [1.0,0.3,0.6,0.7]
#coef_b = [0.5,0.2,0.63,0.77]
#dc = 3.0
#ser_1 = [fourier_series_trunc(dc,coef_a,coef_b,k)]



"""
def fourier_curve_pt(harm_index,harm_amp,)


def fourier_curve()



def cyclic_fourier_seg(t,a,b,phi,psi,M):
    coef_lens = [len(a),len(b),len(phi),len(psi)]
    assert all(x == coef_lens[0] for x in coef_lens), "all coef lists must be of equal length"
    rt = a[0]
    theta_t = tau * M * t
    N = len(a)
    for k in range(1, N):
        rt += a[k] * cos(tau*k*t + phi[k])
        theta_t += b[k] * sin(tau*k*t + psi[k])
    gamma_t = rt * cmath.exp(1j*theta_t)
    #point = [1*gamma_t.real,1*gamma_t.imag]
    return gamma_t

def cyclic_fourier(a,b,phi,psi,M,L,res,**kwargs):
    tv = np.linspace(0, 1, res)
    segments = []
    for k in range(L):
        rot = cmath.exp(1j * tau * k / L)
        arr = []
        for t in tv:
            v = rot * cyclic_fourier_seg(t,a,b,phi,psi,M)
            pt = np.array([1*v.real,1*v.imag])
            arr.append(pt)
        segments.append(np.asarray(arr))
    return segments

def tessellate_fourier_segments(a,b,phi,psi,M,L,res,**kwargs):
    tv = np.linspace(0, 1, res)
    segments = []
    for k in range(L):
        rot = cmath.exp(1j * tau * k / L)
        arr = []
        for t in tv:
            v = rot * cyclic_fourier(t,a,b,phi,psi,M)
            pt = np.array([v.real,v.imag])
            arr.append(pt)
        segments.append(np.asarray(arr))
    return segments

def tessellate_fourier_pts(a,b,phi,psi,M,L,res,**kwargs):
    tv = np.linspace(0, 1, res)
    segments = []
    for k in range(L):
        rot = cmath.exp(1j * tau * k / L)
        arr = []
        for t in tv:
            v = rot * cyclic_fourier(t,a,b,phi,psi,M)
            pt = np.array([v.real,v.imag])
            arr.append(pt)
        segments.append(np.asarray(arr))
    ret = np.asarray(segments).reshape((res*L,2))
    return ret

if __name__ == "__main__":
  from PlotContext import PlotContext
  from tkiter_widgets import IntSlider, NumberListEntry, PresetCtrl
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
  unit_circ = [[cos(a),sin(a)] for a in np.linspace(0,tau,360)]

  pts = cyclic_fourier(**args)

  def plot_pts():
    plt_ctx.clear()
    #plt_ctx.plot_pointlist(unit_circ,"purple",0.3)
    for seg in pts:
        plt_ctx.plot_pointlist(seg,"black", 0.3)


  def arg_change(_id,val):
    global pts
    args[_id] = val
    pts = cyclic_fourier(**args)
    plot_pts()


  def load_preset(data):
    global args, pts
    args = data
    pts = cyclic_fourier(**args)
    plot_pts()

  fp = proj_file_path("/data/fourier_curves.json")
  a_entry = NumberListEntry(plt_ctx,"a","a",args["a"],"float",arg_change)
  b_entry = NumberListEntry(plt_ctx,"b","b",args["b"],"float",arg_change)
  phi_entry = NumberListEntry(plt_ctx,"phi","phi",args["phi"],"float",arg_change)
  psi_entry = NumberListEntry(plt_ctx,"psi","psi",args["psi"],"float",arg_change)
  L_slider = IntSlider(plt_ctx,"L","L",1,10,3,arg_change)
  M_slider = IntSlider(plt_ctx,"M","M",1,10,3,arg_change)
  Preset_ctrl = PresetCtrl(plt_ctx,fp,lambda:args,load_preset)

  plot_pts()
  plt_ctx.run()
"""






