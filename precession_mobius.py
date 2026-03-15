import cmath
from math import cos, sin, pi, tau, radians
from typing import List, Tuple, Sequence
import numpy as np

cmplx_ser = Tuple[float,float] # serialized complex number
mobius_coef = Tuple[complex,complex,complex,complex]
mobius_coef_ser = Tuple[cmplx_ser,cmplx_ser,cmplx_ser,cmplx_ser] #serialized mobius transform coefficents


def serialize_coef(coef: mobius_coef) -> mobius_coef_ser:
    a, b, c, d = coef
    return (a.real, a.imag), (b.real, b.imag), (c.real, c.imag), (d.real, d.imag)

def deserialize_coef(coef:mobius_coef_ser) -> mobius_coef:
    a,b,c,d = coef
    return complex(a[0],a[1]),complex(b[0],b[1]),complex(c[0],c[1]),complex(d[0],d[1])



def mobius_trans(z: complex, a: complex, b: complex, c: complex, d: complex) -> complex:
    """General Möbius transformation"""
    denominator = (c * z) + d
    if denominator == 0:
        return float('inf') * 1j  # Return complex infinity
    return ((a * z) + b) / denominator


def apply_mobius_trans(points:Sequence[complex],coef:mobius_coef):
  a,b,c,d = coef
  return [mobius_trans(p,a,b,c,d) for p in points]


def rotation_to_mobius(alpha: float, theta: float) -> Tuple[complex, complex]:
    """
    Convert sphere rotation to Möbius parameters.
    
    Args:
        alpha: Tilt angle (fixed precession angle) in radians
        theta: Precession angle around vertical axis in radians
    
    Returns:
        Tuple (a, b) such that f(z) = (a*z + b) / (-b.conjugate()*z + a.conjugate())
    """
    #a = cmath.cos(alpha/2) * cmath.exp(1j * theta/2)
    #b = cmath.sin(alpha/2) * cmath.exp(-1j * theta/2)
    a = cmath.cos(alpha/2) * cmath.exp(1j * theta/2)
    b = cmath.sin(alpha/2) * cmath.exp(-1j * theta/2)
    #c = -b.conjugate()
    #d = a.conjugate()
    return a, b

def sphere_rotation_mobius(z: complex, alpha: float, theta: float) -> complex:
    """
    Apply the Möbius transformation corresponding to a sphere rotation.
    
    Args:
        z: Input point in complex plane
        alpha: Tilt angle (fixed precession angle)
        theta: Precession angle around vertical axis
    
    Returns:
        Transformed point
    """
    a, b  = rotation_to_mobius(alpha, theta)
    c = -b.conjugate()
    d = a.conjugate()
    
    return mobius_trans(z, a, b, c, d)

def calc_precession_coefficents(tilt_ang:float,res:int) -> Sequence[mobius_coef]:
  coef = []
  theta_angles = np.linspace(0,tau,360*res)
  for theta in theta_angles:
    a,b = rotation_to_mobius(tilt_ang,theta)
    c = -b.conjugate()
    d = a.conjugate()
    coef.append((a,b,c,d))
  return coef


# Example usage:
if __name__ == "__main__":
    #import matplotlib.pyplot as plt
    from DataFile import DataFile
    from pathlib import Path
    from fourier_curve import angle_and_radial_modulation_curve
    from PlotContext import PlotContext
    from tkiter_widgets import NumberboxInt
    from matplotlib.animation import FuncAnimation, PillowWriter
    #from matplot_anim import create_animation, create_subplot_animation

    BASE_DIR = Path(__file__).resolve().parent
    precession_coefs_file_path = str(BASE_DIR) + "/data/precession_coefs.json"
    precession_coefs_file = DataFile(precession_coefs_file_path)
    fourier_curve_file_path = str(BASE_DIR) + "/data/fourier_curves.json"
    fourier_curve_file = DataFile(fourier_curve_file_path)
    fourier_curve_params = fourier_curve_file.load("curve_3")
    fourier_curve = angle_and_radial_modulation_curve(**fourier_curve_params)
    fourier_curve_cmplx = [complex(p[0],p[1]) for p in fourier_curve]
    disc_radius = fourier_curve_params["disc_radius"]
    precession_coefs = None
    if precession_coefs_file.has_key("tilt1"):
        precession_coefs_serialized = precession_coefs_file.load("tilt1")["coefs"]
        precession_coefs = [deserialize_coef(coef) for coef in precession_coefs_serialized]
    else:
      precession_tilt = radians(22.5)
      precession_coefs = calc_precession_coefficents(precession_tilt,4)
      precession_coefs_serialized = [serialize_coef(coef) for coef in precession_coefs]
      precession_coefs_file.save("tilt1",{"tilt_angle":22.5,"res":4,"coefs":precession_coefs_serialized})

    frames = [apply_mobius_trans(fourier_curve_cmplx,coefs) for coefs in precession_coefs]
    frame_cnt = len(frames)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Your existing code to generate frames...
    # frames = [apply_mobius_trans(fourier_curve_cmplx, coefs) for coefs in precession_coefs]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set up the plot
    disc_radius = 2.0  # or whatever your disc radius is
    padding = 0.2 * disc_radius
    limit = disc_radius + padding

    # Draw the reference circle
    circle = plt.Circle((0, 0), disc_radius, fill=False, color='red', linestyle='--', linewidth=1)
    ax.add_patch(circle)

    # Draw axes
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    # Set up the plot limits
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Create an empty line object that we'll update
    line, = ax.plot([], [], 'b-', linewidth=1.5)

    # Add a title that will show frame number
    title = ax.set_title('Frame 0')


    def init():
        """Initialize the animation with empty data"""
        line.set_data([], [])
        return line, title


    def update(frame_idx):
        """Update the plot for each frame"""
        frame = frames[frame_idx]
        x = [p.real for p in frame]
        y = [p.imag for p in frame]

        line.set_data(x, y)
        title.set_text(f'Frame {frame_idx + 1}/{len(frames)}')

        return line, title


    # Create the animation
    anim = FuncAnimation(
        fig,  # your figure
        update,  # function that updates each frame
        frames=len(frames),  # number of frames
        init_func=init,  # initialization function
        interval=50,  # milliseconds between frames (20 fps)
        blit=True,  # only redraw parts that change (faster)
        repeat=True  # loop when done
    )

    # Show the animation
    plt.show()

    # If you want to control playback:
    # - To stop: close the window
    # - To pause: click the pause button in the matplotlib window
    # - To save: there's usually a save button in the window


    """
    def plot_frame():
      pctx.clear()
      pts = [[p.real,p.imag] for p in frames[args["frame_index"]]]
      pctx.plot_pointlist(pts,"black",0.3)
      pctx.plot_pointlist(circ,"black",0.5)

    def change_frame(_id,val):
      args["frame_index"] = val
      plot_frame()

    frame_index_wid = NumberboxInt(pctx,"frame_index","frame",0,frame_cnt-1,change_frame)
    plot_frame()
    pctx.run()
    """



    # Create standard animation
    """
    create_animation(
        frames=frames,
        disc_radius=disc_radius,
        output_path=str(BASE_DIR) + "/animations/precession1.gif",
        interval=50,  # 50ms between frames = 20 fps
        show_axes=True,
        reference_circle=True
    )

    # Optional: Create subplot animation showing multiple frames simultaneously
    create_subplot_animation(
        frames=frames,
        disc_radius=disc_radius,
        output_path=str(BASE_DIR) + "/animations/precession1_subplots.gif",
        interval=200,  # Slower for subplots
        grid_size=(3, 3)
    )
    """
