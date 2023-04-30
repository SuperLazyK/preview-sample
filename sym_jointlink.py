from sympy import *
from geometry import *
from sym_arith import *

#-------------------------
# Joint/Link
#-------------------------

class RevoluteJoint:
    def __init__(self):
        pass

    def XJ(self, q):
        return Xpln(q, [0, 0])

    def S(self, q):
        return Matrix([1, 0, 0])

    def vJ(self, q, dq):
        return (self.S(q) * dq).reshape(3,1)

    def cJ(self, q, dq):
        # ring(S) == 0
        return zeros(3, 1)


class PrismaticJoint:
    def __init__(self):
        pass

    def XJ(self, q):
        return Xpln(0, [q, 0])

    def S(self, q):
        return Matrix([0, 1, 0])

    def vJ(self, q, dq): # see Ex 4.6
        return (self.S(q) * dq).reshape(3,1)

    def cJ(self, q, dq):
        # ring(S) == 0
        return zeros(3, 1)

class RackPinionJoint:
    def __init__(self, r):
        self.r = r

    def XJ(self, q):
        return Xpln(q, [self.r*q, 0])
        #return Matrix([[1, 0, 0], [q*r*sin(q), cos(q), sin(q)], [q*r*cos(q), -sin(q), cos(q)]])

    def S(self, q):
        return Matrix([1, self.r*cos(q), -self.r*sin(q)])

    def vJ(self, q, dq):
        return Matrix([dq, self.r*cos(q)*dq, -self.r*sin(q)*dq])

    def cJ(self, q, dq):
        return Matrix([0, -self.r*sin(q)*dq**2, -self.r*cos(q)*dq**2])



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
    return Matrix([
        [Ic + m * (cx**2 + cy**2), -m * cy, m * cx],
        [-m * cy, m, 0],
        [m * cx, 0, m]
        ])

# center of mass inertia
def Ic(m, I):
    return Matrix([
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
    def __init__(self, m, I, XT, joint, q, dq):
        self.I = I
        self.XT = XT
        self.joint = joint
        self.m = m
        self.q = q
        self.dq = dq

    def XJ(self):
        return self.joint.XJ(self.q)

    def vJ(self):
        return self.joint.vJ(self.q, self.dq)

    def cJ(self):
        return self.joint.cJ(self.q, self.dq)

    def S(self):
        return self.joint.S(self.q)

    # hmm... not good design
    def drawCmd(self, X_r_to, ctx):
        cmds = []
        for geo in self.geometries(ctx):
            cmds = cmds + geo.drawCmd(X_r_to)
        return cmds

    def force(self):
        return 0

class StickJointLink(JointLink):
    def __init__(self, m, l, joint, q, dq, cx=None, Icog=None, XT=None, tau=0):
        if Icog is None:
            Icog = stickI(m,l)
        if cx is None:
            I = mcI(m, [l/2, 0], Icog)
        else:
            I = mcI(m, [cx, 0], Icog)
        if XT is None:
            XT = Xpln(0, [l, 0])
        super().__init__(m, I, XT, joint, q, dq)
        #self.Ic = Ic(m, Icog)
        self.Xc = Xpln(0, [l/2, 0])
        self.q = q
        self.dq = dq
        self.tau = tau
        self.l = l

    def geometries(self, ctx):
        #return [GeoCircle(0.1*ctx[self.l]), GeoLineSegment(ctx[self.l])]
        l = ctx[self.l] if self.l in ctx else self.l
        return [GeoLineSegment(l)]

    def force(self):
        return self.tau

class StickSpringJointLink(StickJointLink):
    def __init__(self, m, l, k, joint, q, dq):
        super().__init__(m, l, joint, q, dq)
        self.k = k

    def force(self):
        return -self.k * self.q

class WheelJointLink(JointLink):
    def __init__(self, m, r, joint, q, dq, Icog=None, XT=None, tau=0):
        if Icog is None:
            Icog = circleI(m,r)
        I = mcI(m, [0, 0], Icog)
        if XT is None:
            XT = Xpln(0, [0, 0])
        super().__init__(m, I, XT, joint, q, dq)
        #self.Ic = Ic(m, circleI(m,r))
        self.Xc = Xpln(0, [0, 0])
        self.tau = tau
        self.r = r

    def force(self):
        return self.tau

    def geometries(self, ctx):
        r = ctx[self.r] if self.r in ctx else self.r
        return [GeoCircle(r), GeoLineSegment(r)]

