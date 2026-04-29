"""
Special Euclidian Group
"""

import numpy as np

class SE3:
    def __init__(self, q:(float,float,float,float), t):
        self.q = q
        self.t = np.array(t)


    def compose(self, other):
        # rotation
        q_new = quat_mult(self.q, other.q)

        # translate: rotate other's translation into current frame
        t_rot = quat_rotate(self.q, other.t)[1:]
        t_new = self.t + t_rot

        return SE3(q_new, t_new)

    @staticmethod
    def identity():
        return SE3((1.0, 0.0, 0.0, 0.0), np.zeros(3))








