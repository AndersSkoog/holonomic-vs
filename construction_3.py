import numpy as np
from holonomic_view import holonomic_view_3
from holonomic_roll import rolltranslation, inital_R
from sphere_curves import random_closed_sphere_curve
from constants import MIN_VAL


def in_unit_disc(p):
 x,y = p[0],p[1]
 return (x*x)+(y*y) <= 1

def to_complex(p):
 x,y = p
 return complex(x,y)



def extract_holonomic_roll_data(sphere_curve):
  prev_contact = np.array([0.0,0.0,0.0])
  prev_R = inital_R(sphere_curve)
  init_tor_pt = prev_R @ np.array([0.0,0.0,1.0])
  contact_curve = []
  orients = []
  torsion_curve = []

  for i in range(len(sphere_curve[0])-2):
    contact,R = rolltranslation(sphere_curve=sphere_curve,index=i,contact=prev_contact,R=prev_R)
    tor_curve_z = R @ np.array([0.0,0.0,1.0])
    tor_curve_pt = np.array([contact[0],contact[1],tor_curve_z[2]])
    contact_curve.append(contact)
    orients.append(R)
    torsion_curve.append(tor_curve_pt)
    prev_contact=contact
    prev_R=R

  return contact_curve,orients,torsion_curve


def complex_disc_curve(contact_curve):
  out = []
  for p in contact_curve:
    z = to_complex(p)
    if in_unit_disc(p): out.append(z)
    else: out.append(1/z)
  return out


if __name__ == "__main__":
  import pyqtgraph as pg
  from pyqtgraph.Qt import QtWidgets
  contact_curve,orients,torsion_curve = extract_holonomic_roll_data(random_closed_sphere_curve())

  app = pg.mkQApp()
  win = QtWidgets.QWidget()
  layout = QtWidgets.QVBoxLayout()
  win.setLayout(layout)

  plot_widget = pg.GraphicsLayoutWidget()
  layout.addWidget(plot_widget)

  plot = plot_widget.addPlot()
  scatter = pg.ScatterPlotItem(size=3)
  plot.addItem(scatter)

  slider = QtWidgets.QSlider()
  slider.setOrientation(pg.QtCore.Qt.Horizontal)

  def update_index(i):
    view_obj = holonomic_view_3(contact_curve, i)
    pts = np.array(view_obj["view_points"])
    scatter.setData(pos=pts)

  slider.valueChanged.connect(update_index)
  app.exec()
  win.show()










