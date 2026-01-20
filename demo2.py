

if __name__ == "__main__":
 from PlotContext import PlotContext
 from fourier_curve import cyclic_fourier_curve
 from projection import plane_to_sphere
 from tkiter_widgets import NumberboxInt
 from hopf import hopf_fiber_from_s2, hopf_fiber_from_plane_curve_pt, hopf_map
 from S2 import R3_to_S2
 import numpy as np

 curve_args_2 = {
    "M": 1,
    "L": 10,
    "res": 360,
    "ca": [0.0, 0.1, 0.1, 0.1],
    "cb": [0.0, 0.25, 0.08, 0.03],
    "cphi": [0.0, 0.2, -0.5, 0.1],
    "cpsi": [0.0, 0.0, 0.3, -0.2]
 }

 curve_args = {
     "M": 4,
     "L": 2,
     "res": 360,
     "ca": [1.0,-1.0,0.0,0.0],
     "cb": [0.5,6.0,2.0,2.0],
     "cphi": [1.4,0.31,0.31,0.1],
     "cpsi": [1.0,6.2,4.0,-4.2]
 }
 sphere_pts = [
     [1,np.deg2rad(0),np.deg2rad(15)],
     [1,np.deg2rad(10),np.deg2rad(25)],
     [1,np.deg2rad(20),np.deg2rad(35)],
     [1, np.deg2rad(30), np.deg2rad(45)],
     [1, np.deg2rad(40), np.deg2rad(55)],
     [1, np.deg2rad(50), np.deg2rad(65)],
     [1, np.deg2rad(60), np.deg2rad(75)],
     [1, np.deg2rad(70), np.deg2rad(85)],
     [1, np.deg2rad(80), np.deg2rad(90)]
 ]

 def proj_fiber(s2):
   fiber = hopf_fiber_from_s2(s2)
   return [hopf_map(fp) for fp in fiber]

 #test_plt_pts = [proj_fiber(s2) for s2 in sphere_pts]

 #test_plt_pts = [proj_fiber(s2) for s2 in sphere_pts]
 #test_c2 = [hopf_fiber_from_s2(p) for p in sphere_pts]
 #test_r3 = [hopf_map(p) for p in test_c2]
 cpts = cyclic_fourier_curve(**curve_args_2)
 scpts = [plane_to_sphere(p,1) for p in cpts]
 plt_pts = [proj_fiber(p) for p in scpts[0:200]]

 pctx = PlotContext(-0.5,0.5,"hopf map", proj="3d")
 pctx.plot_pointlists(plt_pts,"black",0.2)
 #fibers = [hopf_fiber_from_plane_curve_pt(cpts,i) for i in range(len(cpts))]
 #plt_pts = [[hopf_map(fp) for fp in fiber] for fiber in fibers]
 #pctx.plot_pointlists(plt_pts,"black",0.2)

 #for cl in tc: pctx.plot_pointlists(cl,"black",0.2)
 #def curve_pos_change(id_,val):
 #  fiber = hopf_fiber_from_plane_curve_pt(cpts,val)
 #  plt_pts = [hopf_map(fp) for fp in fiber]
 #  pctx.plot_pointlist(plt_pts)

 #curve_pos = NumberboxInt(pctx,"curve_pos","curve_pos",curve_pos_change,21,len(cpts))
 pctx.run()
