import numpy as np
import cmath

def stereo_proj(z1:complex,z2:complex):
  denom = 1 - z2.real  # or z2 component along projection axis
  x = z1.real / denom
  y = z1.imag / denom
  z = z2.imag / denom
  return np.array([x,y,z])

def hopf_fiber(theta,phi,tv):
  z1 = lambda t: cmath.cos(phi/2) * cmath.exp(1j*t)
  z2 = lambda t: cmath.sin(phi/2) * cmath.exp(1j*(theta+t))
  fiber = np.array([(z1(t),z2(t)) for t in tv])
  return fiber

def proj_hopf_link(theta_1,theta_2,phi_1,phi_2,tv,**kwargs):
  fiber1 = hopf_fiber(theta_1, phi_1, tv)
  fiber2 = hopf_fiber(theta_2, phi_2, tv)
  ret = []
  ret.extend([stereo_proj(f[0], f[1]) for f in fiber1])
  ret.extend([stereo_proj(f[0], f[1]) for f in fiber2])
  return ret

##------------------DEMO-------------------------------------------

if __name__ == "__main__":
  from tkiter_widgets import FloatSlider
  from PlotContext import PlotContext
  from math import tau

  args = {"theta_1":0.1,"phi_1":0.9,"theta_2":0.2,"phi_2":0.7,"tv":np.linspace(0,tau,360)}
  pctx = PlotContext(-1,1,"hopf link projection demo",proj="3d")

  def wid_change(_id, val):
      args[_id] = val
      ret = proj_hopf_link(**args)
      pctx.clear()
      pctx.plot_pointlist(ret, "black", 0.5)


  theta_slider_1 = FloatSlider(pctx, "theta_1", "theta_1", 0.0, tau, args["theta_1"], wid_change)
  phi_slider_1 = FloatSlider(pctx, "phi1_", "phi_1", 0.0, tau, args["phi_1"], wid_change)
  theta_slider_2 = FloatSlider(pctx, "theta_2", "theta_2", 0.0, tau, args["theta_2"], wid_change)
  phi_slider_2 = FloatSlider(pctx, "phi_2", "phi_2", 0.0, tau, args["phi_2"], wid_change)
  pctx.run()

