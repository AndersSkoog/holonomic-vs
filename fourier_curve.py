from math import tau, cos, sin
import numpy as np
import cmath
from DataFile import DataFile, BASE_DIR

def cyclic_fourier(t:float,a,b,phi,psi,M) -> complex:
    coef_lens = [len(a),len(b),len(phi),len(psi)]
    assert all(x == coef_lens[0] for x in coef_lens), "all coef lists must be of equal length"
    rt = a[0]
    theta_t = tau * M * t
    N = len(a)
    for k in range(1, N):
        rt += a[k] * cos(tau*k*t + phi[k])
        theta_t += b[k] * sin(tau*k*t + psi[k])
    gamma_t = rt * cmath.exp(1j*theta_t)
    #point = [gamma_t.real,gamma_t.imag]
    return gamma_t

def tessellate_fourier_pts(a,b,phi,psi,M,L,res,**kwargs):
    tv = np.linspace(0, 1, res)
    ret = []
    for k in range(L):
        rot = cmath.exp(1j * tau * k / L)
        arr = []
        for t in tv:
            v = rot * cyclic_fourier(t,a,b,phi,psi,M)
            pt = np.array([v.real,v.imag])
            arr.append(pt)
        ret.append(arr)
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
  pts = tessellate_fourier_pts(**args)

  def plot_pts():
    plt_ctx.clear()
    for arr in pts:
      plt_ctx.plot_pointlist(arr, "black", 0.3)


  def arg_change(_id,val):
    global pts
    args[_id] = val
    pts = tessellate_fourier_pts(**args)
    plot_pts()


  def load_preset(data):
    print(data)

  fp = str(BASE_DIR) + "/data/fourier_curves.json"
  print(fp)
  a_entry = NumberListEntry(plt_ctx,"a","a",args["a"],"float",arg_change)
  b_entry = NumberListEntry(plt_ctx,"b","b",args["b"],"float",arg_change)
  phi_entry = NumberListEntry(plt_ctx,"phi","phi",args["phi"],"float",arg_change)
  psi_entry = NumberListEntry(plt_ctx,"psi","psi",args["psi"],"float",arg_change)
  L_slider = IntSlider(plt_ctx,"L","L",1,10,3,arg_change)
  M_slider = IntSlider(plt_ctx,"M","M",1,10,3,arg_change)
  Preset_ctrl = PresetCtrl(plt_ctx,fp,lambda:args,load_preset)

  plot_pts()
  plt_ctx.run()







