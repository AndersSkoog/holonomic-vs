import numpy as np

def bezier_curve(control_pts, t):
    pts = np.array(control_pts, dtype=float)
    for _ in range(len(pts) - 1):
        pts = (1 - t) * pts[:-1] + t * pts[1:]
    return pts[0]

def sample_bezier(control_pts, n=200):
    ts = np.linspace(0.0, 1.0, n)
    return np.array([bezier_curve(control_pts, t) for t in ts])


def cubic_bezier(P0, P1, P2, P3, t):
    return (
        (1 - t)**3 * P0 +
        3 * (1 - t)**2 * t * P1 +
        3 * (1 - t) * t**2 * P2 +
        t**3 * P3
    )

def sample_cubic(P0, P1, P2, P3, n=200):
    ts = np.linspace(0.0, 1.0, n)
    return np.array([cubic_bezier(P0, P1, P2, P3, t) for t in ts])