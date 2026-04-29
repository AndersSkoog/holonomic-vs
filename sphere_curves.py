import numpy as np


def random_closed_sphere_curve_dirs(n=360, k=5, seed=None):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    theta = np.zeros(n)
    phi = np.zeros(n)

    for i in range(1, k+1):               # sum over harmonics
        # random amplitudes and phases for theta and phi separately
        A_theta, B_theta = rng.normal(size=2)
        A_phi, B_phi = rng.normal(size=2)
        phase_theta = rng.uniform(0, 2*np.pi)
        phase_phi = rng.uniform(0, 2*np.pi)

        theta += A_theta * np.cos(i*t + phase_theta) + B_theta * np.sin(i*t + phase_theta)
        phi   += A_phi   * np.cos(i*t + phase_phi)   + B_phi   * np.sin(i*t + phase_phi)

    return theta,phi

def random_closed_sphere_curve(n=360, k=5, radius=1.0, seed=None):
    theta,phi = random_closed_sphere_curve_dirs
    print(theta)
    print(phi)
    # Convert to Cartesian coordinates on the sphere
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    pts = [np.array(x[i],y[i],z[i]) for i in range(n)]
    dirs = [(theta[i],phi[i]) for i in range(n)]
    return pts,dirs


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    pts,dirs = random_closed_sphere_curve(n=360, k=5, radius=1.0, seed=42)
    xs,ys,zs = zip(*pts)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, 'b-', linewidth=1)
    ax.set_aspect('equal')
    plt.show()





