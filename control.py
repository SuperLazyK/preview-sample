from linktree import *
import numpy as np
from jointlink import *
import sys

g=9.8

def diff(xs, n, dt):
    lastn = xs[-(n+1):]
    if n == 1:
        return (lastn[1] - lastn[0]) / dt
    elif n == 2:
        return (lastn[2] + lastn[0] - 2*lastn[1]) / (dt**2)
    else:
        assert False, "not implemented"

class SimpleRWPController():
    def __init__(self):
        self.Ls = [0, 0, 0]
        self.qcs = [0, 0, 0]
        self.Tcs = []

    def tau(self, qc, dt, fict_model):
        H = fict_model.composite_inertia()
        jl1 = fict_model.jointlinks[1]
        jl2 = fict_model.jointlinks[2]

        # observe
        dq1 = jl1.dq
        dq2 = jl2.dq
        L = H[1][1] * dq1 + H[1][2] * dq2
        self.Ls.append(L)
        self.qcs.append(qc)

        # control
        Ls = self.Ls
        qcs = self.qcs
        L = Ls[-1]
        qc = qcs[-1]
        dL = diff(Ls, 1, dt)
        ddL = diff(Ls, 2, dt)
        dqc = diff(qcs, 1, dt)
        ddqc = diff(qcs, 2, dt)
        q2 = jl2.q
        H2 = H.copy()
        H2[0][0] = 0
        H2[1][0] = 0
        H2[2][0] = -1
        C = fict_model.inverse_dynamics(np.zeros(fict_model.NB), np.zeros((fict_model.NB, fict_model.dim)))
        D = H[1][2] * H[0][1] - H[1][1] * H[0][2]
        assert D != 0
        Y1 = H[0][1] / D
        Y2 = H[1][1] / D / g
        Tc = -H[1][1] / H[0][1] / g
        self.Tcs.append(Tc)
        lmd1 = -1. / Tc
        lmd2 = -20 # [rad/sec]
        lmd3 = -20 # [rad/sec]
        lmd4 = -20 # [rad/sec]
        a0 = lmd1 * lmd2 * lmd3 * lmd4
        a1 = - (lmd1 * lmd2 * lmd3 + lmd1 * lmd2 * lmd4 + lmd1 * lmd3 * lmd4 + lmd2 * lmd3 * lmd4)
        a2 = (lmd1 * lmd2 + lmd1 * lmd3 + lmd1 * lmd4 + lmd2 * lmd3 + lmd2 * lmd4 + lmd3 * lmd4)
        a3 = - (lmd1 + lmd2 + lmd3 + lmd4)
        kdd = -a3
        kd = -a2 + a0 * Y2 / Y1
        kL = -a1
        kq = -a0 / Y1
        alpha2 = 1 / (lmd3 * lmd4)
        alpha1 = alpha2 * (lmd3 + lmd4)
        u = qc + alpha1 * dqc + alpha2 * ddqc
        dddL =  kdd * ddL + kd * dL + kL * L + kq * (q2 - u)
        tau2 = np.linalg.solve(H2, -C - np.array([dddL /g, 0, 0]))[0]
        return tau2


def test_fictitious_link():
    jl0 = StickJointLink(0, 0, PrismaticJoint())
    jl1 = StickJointLink(1, 1, RevoluteJoint(), q=np.pi/2*0.9)
    jl2 = WheelJointLink(1, 0.5, RevoluteJoint())
    fict_model = LinkTreeModel([jl0, jl1, jl2], g_acc=g)
    plant_model = LinkTreeModel([jl1, jl2], g_acc=g)
    controller = SimpleRWPController()

    import graphic
    viewer = graphic.Viewer(scale=100, offset=[0, -0.1])
    t = 0
    while True:
        dt = 0.001
        qc = 0
        tau2 = controller.tau(qc, dt, fict_model)
        jl2.tau = tau2
        plant_model.step(dt)
        t = t + dt

        viewer.clear()
        viewer.handle_event(graphic.default_input_handler)
        viewer.text([f"q1:{jl1.q:.03f} q2:{jl2.q:.03f}",
            f"t: {t:.03f} tau2:{tau2:.03f}"])
        viewer.draw(plant_model.drawCmds())
        viewer.flush()


if __name__ == '__main__':
    test_fictitious_link()

