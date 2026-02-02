from math import tau, cos, sin
import cmath

def cyclic_fourier_curve(ca,cb,cphi,cpsi,M,L,res,**kwargs):
    lens = [len(ca),len(cb),len(cphi),len(cpsi)]
    hcnt = len(ca)
    assert all(x == hcnt for x in lens), "all coef lists must be of equal length"
    indicies = range(0,res * L)
    pts = []
    for i in indicies:
      vi = i % res
      if vi == 0: continue
      Li = int(i/res)
      t = (tau/res) * vi
      rot = cmath.exp(1j * tau * Li / L)
      r_t = ca[0]
      theta_t = tau * M * t
      for k in range(hcnt):
        r_t += ca[k] * cos(tau*k*t + cphi[k])
        theta_t += cb[k] * sin(tau*k*t + cpsi[k])
      gamma_t = r_t * cmath.exp(1j*theta_t)
      gamma_rot = rot * gamma_t
      pts.append([gamma_rot.real,gamma_rot.imag])
    return pts


if __name__ == "__main__":
  from PlotContext import PlotContext
  from tkiter_widgets import FloatSlider, IntSlider, NumberListEntry
  args = {
      "ca":[0.0,0.0,0.1,0.0],
      "cb":[0.5, 0.0, 0.0, 0.0],
      "cphi":[0.0, 0.0, 0.1, 0.0],
      "cpsi":[0.5, 0.0, 0.0, 0.0],
      "L":2,
      "M":3,
      "res":360
  }
  plt_ctx = PlotContext(-1,1,"cyclic fourier curve",proj="2d")

  def arg_change(_id,val):
    args[_id] = val
    plt_ctx.clear()


    plt_ctx.plot_pointlist(cyclic_fourier_curve(**args),lw=0.2)


  ca_entry = NumberListEntry(plt_ctx,"ca","ca",args["ca"],"float",arg_change)
  cb_entry = NumberListEntry(plt_ctx,"cb","cb",args["cb"],"float",arg_change)
  cphi_entry = NumberListEntry(plt_ctx,"cphi","cphi",args["cphi"],"float",arg_change)
  cpsi_entry = NumberListEntry(plt_ctx,"cpsi","cpsi",args["cpsi"],"float",arg_change)
  L_slider = IntSlider(plt_ctx,"L","L",1,10,3,arg_change)
  M_slider = IntSlider(plt_ctx,"M","M",1,10,3,arg_change)

  plt_ctx.plot_pointlist(cyclic_fourier_curve(**args), lw=0.2)
  plt_ctx.run()







