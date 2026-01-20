import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def detect_shape(data):
    shape = np.shape(data)
    if len(shape) == 2 and shape[1] == 2: return "pointlist"
    if len(shape) == 3 and shape[1:] == (2,2): return "lines"
    if len(shape) == 3 and shape[2] == 2: return "pointlists"
    raise ValueError(f"Cannot interpret shape {shape}")

class PlotContext:
    def __init__(self, dmin,dmax,title="plot",proj="2d"):
        assert proj == "2d" or "3d", "not valid projection argument"
        self.proj = proj
        self.root = tk.Tk()
        self.root.title(title)
        # left = widgets, right = plot
        self.widget_frame = ttk.Frame(self.root)
        self.widget_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.fig = None
        self.ax = None
        if proj == "2d":
            self.fig, self.ax = plt.subplots()
            plt.xlim((dmin,dmax))
            plt.ylim((dmin, dmax))
            self.ax.set_autoscale_on(False)
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1,projection="3d")
            self.ax.set_xlim(dmin, dmax)
            self.ax.set_ylim(dmin, dmax)
            self.ax.set_zlim(dmin, dmax)
            self.ax.set_autoscale_on(False)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False)


    def clear(self):
        self.ax.clear()
        self.ax.set_aspect("equal", "box")
        self.canvas.draw()

    def plot_pointlist(self, pts, col="black", lw=0.2):
        if self.proj == "2d":
            x,y = zip(*pts)
            self.ax.plot(x,y,color=col,linewidth=lw)
        if self.proj == "3d":
            x,y,z = zip(*pts)
            self.ax.plot(x,y,z,color=col,linewidth=lw)
        self.canvas.draw()

    def plot_marker(self,pt,size,color):
        self.ax.plot(pt[0],pt[1],pt[2],color=color,marker="x",markersize=size)

    def plot_pointlists(self,ptsl,col,lw):
        for i in range(len(ptsl)):
            self.plot_pointlist(ptsl[i],col,lw)

    def plot_lines(self, lines, col="black", lw=1):
        lines_shape = np.shape(lines)
        pred = [len(lines_shape) == 3,lines_shape[1] == 2,lines_shape[2] in [2,3]]
        assert all(pred), "shape of line must be (x,2,2) or (x,2,3)"
        line_dim = lines_shape[2]
        if line_dim == 2:
            for (p1,p2) in lines:
              self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]],color=col, linewidth=lw)
            self.canvas.draw()
        if line_dim == 3:
            for (p1,p2) in lines:
              self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=col, linewidth=lw)
            self.canvas.draw()


    def run(self):
        self.root.mainloop()

