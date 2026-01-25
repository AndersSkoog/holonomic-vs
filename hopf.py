import numpy as np
import cmath

from S2 import stereo_project_R2_R3,R3_to_S2
from plane_torision import torsion_angle
from SO import SO_3
from lib import angles
from typing import Sequence, TypedDict, NamedTuple

def fiber_pt(theta: float, phi: float, t: float) -> (complex,complex):
  e_it = cmath.exp(1j * t)
  z1 = cmath.cos(phi/2) * e_it
  z2 = cmath.sin(phi/2) * cmath.exp(1j*(t + theta))
  return z1, z2

def fiber(theta:float,phi:float,tv) -> Sequence[(complex,complex)]:
  return np.asarray([fiber_pt(theta,phi,t) for t in tv])

def fiber_pt_to_quaternion(theta:float,phi:float,t:float) -> Sequence[float]:
  z1,z2 = fiber_pt(theta,phi,t)
  w = z1.real
  x = z1.imag
  y = z2.real
  z = z2.imag
  # optionally renormalize to avoid numerical drift
  norm = np.sqrt(w*w + x*x + y*y + z*z)
  if norm == 0:
    return 1.0, 0.0, 0.0, 0.0
  return [w / norm, x / norm, y / norm, z / norm]

def proj_fiber_pt(theta:float,phi:float,t:float) -> Sequence[float]:
  z1,z2 = fiber_pt(theta,phi,t)
  denom = 1 - z2.real  # or z2 component along projection axis
  x = z1.real / denom
  y = z1.imag / denom
  z = z2.imag / denom
  return np.asarray([x,y,z])

def proj_fiber_pt_2(c2:(complex,complex)) -> Sequence[float]:
  z1,z2 = c2
  denom = 1 - z2.real  # or z2 component along projection axis
  x = z1.real / denom
  y = z1.imag / denom
  z = z2.imag / denom
  return np.asarray([x,y,z])

def proj_hopf_link(theta_1,theta_2,phi_1,phi_2,tv,**kwargs) -> (Sequence[float],Sequence[float]):
  circ_1 = np.asarray([proj_fiber_pt(theta_1,phi_1,t) for t in tv])
  circ_2 = np.asarray([proj_fiber_pt(theta_2,phi_2,t) for t in tv])
  return circ_1,circ_2


def proj_hopf_link_2(fiber1:Sequence[(complex,complex)],fiber2:Sequence[(complex,complex)]) -> (Sequence[float],Sequence[float]):
  circ_1 = np.asarray([proj_fiber_pt_2(c2) for c2 in fiber1])
  circ_2 = np.asarray([proj_fiber_pt_2(c2) for c2 in fiber2])
  return circ_1,circ_2


def hopf_link_from_disc_pt(disc_pts:Sequence[float],index:int,R=1):
  li = len(disc_pts) - 1
  assert 0 <= index < li, "index out of range"
  i1 = index
  i2 = 0 if index == li else index + 1
  p1, p2 = disc_pts[i1], disc_pts[i2]
  px, py = p1
  ta = torsion_angle(p1, p2)
  r3_1 = stereo_project_R2_R3([px,py],R)
  s2_1 = R3_to_S2(r3_1,R)
  th1,ph1 = s2_1[1],s2_1[2]
  so3 = SO_3(th1,ph1,ta)
  r3_2 = r3_1 @ so3.T
  s2_2 = R3_to_S2(r3_2,R)
  th2, ph2 = s2_2[1],s2_2[2]
  fiber1 = fiber(th1,ph1,angles)
  fiber2 = fiber(th2,ph2,angles)
  circ1,circ2 = proj_hopf_link_2(fiber1,fiber2)
  return {"fiber1":fiber1,"fiber2":fiber2,"circ1":circ1,"circ2":circ2}



##------------------DEMO-------------------------------------------

if __name__ == "__main__":
  from tkiter_widgets import FloatSlider
  from PlotContext import PlotContext
  from math import tau

  args = {"theta_1":0.1,"phi_1":0.9,"theta_2":0.2,"phi_2":0.7,"tv":np.linspace(0,tau,360)}
  pctx = PlotContext(-1,1,"hopf link projection demo",proj="3d")

  def wid_change(_id, val):
      args[_id] = val
      circ1,circ2 = proj_hopf_link(**args)
      pctx.clear()
      pctx.plot_pointlist(circ1,"black",0.5)
      pctx.plot_pointlist(circ2, "black", 0.5)


  theta_slider_1 = FloatSlider(pctx, "theta_1", "theta_1", 0.0, tau, args["theta_1"], wid_change)
  phi_slider_1 = FloatSlider(pctx, "phi_1", "phi_1", 0.0, tau, args["phi_1"], wid_change)
  theta_slider_2 = FloatSlider(pctx, "theta_2", "theta_2", 0.0, tau, args["theta_2"], wid_change)
  phi_slider_2 = FloatSlider(pctx, "phi_2", "phi_2", 0.0, tau, args["phi_2"], wid_change)
  pctx.run()

