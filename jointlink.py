import numpy as np
from numpy import sin, cos, abs
from geometry import *
from arith import *

#-------------------------
# Joint/Link
#-------------------------

class RevoluteJoint:
    def __init__(self):
        pass

    def XJ(self, q):
        return Xpln(q, [0, 0])

    def S(self):
        return np.array([1, 0, 0])

    def T(self):
        return np.array([[0, 1, 0 ], [0, 0, 1]]).T

    def vJ(self, qd):
        return (self.S() * qd).reshape(-1) # @

    def cJ(self, qd):
        # ring(S) == 0
        return np.zeros(3)


class PrismaticJoint:
    def __init__(self):
        pass

    def XJ(self, q):
        return Xpln(0, [q, 0])

    def S(self):
        return np.array([0, 1, 0])

    def T(self):
        return np.array([[1, 0, 0 ], [0, 0, 1]]).T

    def vJ(self, qd): # see Ex 4.6
        return (self.S() * qd).reshape(-1) # @

    def cJ(self, qd):
        # ring(S) == 0
        return np.zeros(3)


# planar inertia: M -> F
# c : cog position on the body coordinate Fi
# body coordinate 
# note:
#  - center coordinate inertia tensor: diag(Ic, m, m)
#  - X_body_to_center : Xpln(0, c)
#  - X_center_to_dual : inv(Xpln(0, c))
#  see (2.62~2.63)
#  Ibody =  dual(inv(Xplan(0,c))) diag(Ic, m, m) Xplan(0,c) vbody
#  note dual(inv(X)) = X^T
def mcI(m, c, Ic):
    cx = c[0]
    cy = c[1]
    return np.array([
        [Ic + m * (cx**2 + cy**2), -m * cy, m * cx],
        [-m * cy, m, 0],
        [m * cx, 0, m]
        ])

def Ic(m, I):
    return np.array([
        [I, 0, 0],
        [0, m, 0],
        [0, 0, m]
        ])

def stickI(m, l):
    return m * l**2 / 12

def circleI(m, r):
    return r**2 * m / 2

DIM=3

# kinematics tree element
# body with a joint
# index -1 means root
class JointLink():
    # note:
    # Link and joint attached to the previous link
    # current body coordinate' origin is at the previous joint position
    # the next joint is attached at XT on the current body coordinate
    # dim == 1 because 2-dim joint can decomposed into 2 joints
    def __init__(self, m, I, XT, joint):
        self.I = I
        self.XT = XT
        self.joint = joint
        self.geometries = []
        self.m = m
        self.q = 0
        self.dq = 0
        self.tau = 0
        self.X_r_to = np.eye(DIM)
        self.vel = 0 # vel in body coordinate

    def XJ(self):
        return self.joint.XJ(self.q)

    def vJ(self):
        return self.joint.vJ(self.dq)

    def cJ(self):
        return self.joint.cJ(self.dq)

    def S(self):
        return self.joint.S()

    def drawCmd(self):
        cmds = []
        for geo in self.geometries:
            cmds = cmds + geo.drawCmd(self.X_r_to)
        return cmds

class StickJointLink(JointLink):
    def __init__(self, m, l, joint, q=0, dq=0):
        I = mcI(m, [l/2, 0], stickI(m,l))
        XT = Xpln(0, [l, 0])
        super().__init__(m, I, XT, joint)
        self.Ic = Ic(m, stickI(m,l))
        self.geometries = [GeoCircle(0.1*l), GeoLineSegment(l)]
        self.Xc = Xpln(0, [l/2, 0])
        self.q = q
        self.dq = dq

    def active_force(self):
        return self.tau

class StickSpringJointLink(StickJointLink):
    def __init__(self, m, l, k, joint, q=0, dq=0):
        super().__init__(m, l, joint, q, dq)
        self.k = k

    def active_force(self):
        return -self.k * self.q

class WheelJointLink(JointLink):
    def __init__(self, m, r, joint, q=0, dq=0):
        I = mcI(m, [0, 0], circleI(m,r))
        XT = Xpln(0, [0, 0])
        super().__init__(m, I, XT, joint)
        self.Ic = Ic(m, circleI(m,r))
        self.geometries = [GeoCircle(r)]
        self.Xc = Xpln(0, [0, 0])

    def active_force(self):
        return self.tau


