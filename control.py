from linktree import *
import numpy as np
from jointlink import *
import sys
import time
from arith import *
from signal_proc import *

#rwp = True
rwp = False

# NOTE:
# q1 is passive
# q2 is active (qc)
# q0 is fict
class Simple2RRWPController():
    def __init__(self, g_acc=9.81):
        self.g_acc = g_acc
        self.Ls = [0, 0, 0]
        self.qcs = []
        self.qfs = []
        self.ts = []

    def tau(self, t, qc, dt, fict_model, previewT = 0):
        H = fict_model.composite_inertia()
        jl1 = fict_model.jointlinks[1]
        jl2 = fict_model.jointlinks[2]
        #print("H", H)
        assert H[1][1] / H[0][1] < 0
        Tc = np.sqrt(-H[1][1] / H[0][1] / self.g_acc)

        # observe
        dq1 = jl1.dq
        dq2 = jl2.dq
        L = H[1][1] * dq1 + H[1][2] * dq2
        print("t:", t)
        print("qc", qc)
        print("q1", jl1.q)
        print("q2", jl2.q)
        print("L", L)
        self.ts.append(t)
        self.Ls.append(L)

        # control
        Ls = self.Ls
        if previewT >= dt:
            N = int(previewT / dt)
            while len(self.qcs) < N:
                self.qcs.append(qc)
            self.qcs.append(qc)
            qf = rlpf(self.qcs[-(N+1):], Tc, dt)
            while len(self.qfs) < 2:
                self.qfs.append(qf)
                self.qfs.append(qf)
            self.qfs.append(qf)
            qcs = self.qfs
            #print(qcs)
        else:
            if len(self.qcs) == 0:
                self.qcs.append(qc)
                self.qcs.append(qc)
            self.qcs.append(qc)
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
        Y2 = H[1][1] / D / self.g_acc
        #print("H", H)
        #print("D", D)
        #print("Y1", Y1)
        #print("Y2", Y2)
        #print("Tc", Tc)
        I,m,cog = fict_model.syn_inertia_info()
        #print("Gv", -D/(m*H[1,1]))
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
        alpha1 = -alpha2 * (lmd3 + lmd4)
        u = qc + alpha1 * dqc + alpha2 * ddqc
        print(f"u: {u:.03f} = {qc:.03f} + {alpha1:.03f} * {dqc:.03f} + {alpha2:.03f} * {ddqc:.03f}")
        dddL =  kdd * ddL + kd * dL + kL * L + kq * (q2 - u)
        print(f"dddL: {dddL:.01f} = {kdd * ddL:.01f}({kdd:.01f} * {ddL:.01f}) + {kd * dL:.01f}({kd:.01f} * {dL:.01f}) + {kL * L:.01f}({kL:.01f} * {L:.01f}) + {kq * (q2 - u):.01f}({kq:.01f} * ({q2:.01f} - {u:.01f}))")
        tau2 = np.linalg.solve(H2, -C - np.array([dddL /self.g_acc, 0, 0]))[0]
        print("tau", tau2)
        return tau2

def sample_input(dt = 0.001):
    pfs = [ (lambda t:0, [0,1])
          , (lambda t:2, [1,2.25])
          , (lambda t:0, [2.25,2.75])
          , (lambda t:2, [2.75,3.75])
          , (lambda t:2 -6*(t-3.75), [3.75,4.25])
          , (lambda t:-1, [4.25, 5])
          , (lambda t:-1+2*(t-5), [5, 6])
          , (lambda t:1-(t-6), [6, 7])
          , (lambda t:0, [7, 7.5])
          , (lambda t:sin(2*np.pi*(t-7.5)), [7.5, 9.5])
          , (lambda t:0, [9.5, 10])
          ]
    t = 0
    ts = []
    xs = []
    while t <10:
        x = sample(pfs, t)
        xs.append(x)
        ts.append(t)
        t = t + dt
    return ts, xs


def models():
    g_acc=9.81
    jl0 = StickJointLink(0, 0, PrismaticJoint())
    if rwp:
        jl1 = StickJointLink(1, 1, RevoluteJoint(), Icog=0, XT=Xpln(0, [0.5, 0]), q=np.pi/2)
        jl2 = WheelJointLink(1, 0.5, RevoluteJoint())
    else:
        jl1 = StickJointLink(0.7, 0.2, RevoluteJoint(), cx = 0.2, Icog=0, q=np.pi/2)
        cx = (0.25*0.5 + (0.25+0.35)*0.3)/0.8
        Icog = (cx - 0.25)**2 * 0.5 + (cx-0.6)**2 * 0.3
        jl2 = StickJointLink(0.8, cx, RevoluteJoint(), cx = cx, Icog=Icog)

    #jl1 = StickJointLink(1, 1, RevoluteJoint(), Icog=0, XT=Xpln(0, [0.5, 0]), q=np.pi/2)
    #jl2 = WheelJointLink(1, 1, RevoluteJoint())
    fict_model = LinkTreeModel([jl0, jl1, jl2], g_acc=g_acc)
    plant_model = LinkTreeModel([jl1, jl2], g_acc=g_acc)
    controller = Simple2RRWPController()
    return controller, plant_model, fict_model, jl1, jl2



import sympy
r, l, m1, m2, g = sympy.symbols('r l m1 m2 g')
#r=0.5
#r=1
#l=1
#m1=1
#m2=1

def sym_models():
    jl0 = StickJointLink(0, 0, PrismaticJoint())
    if rwp:
        jl1 = StickJointLink(m1, l, RevoluteJoint(), Icog=0, XT=Xpln(0, [0.5, 0]), q=np.pi/2)
        jl2 = WheelJointLink(m2, r, RevoluteJoint())
    else:
        jl1 = StickJointLink(m1, l, RevoluteJoint(), Icog=0, q=np.pi/2)
        jl2 = StickJointLink(m2, r, RevoluteJoint(), Icog=0)
    fict_model = LinkTreeModel([jl0, jl1, jl2], g_acc=g)
    plant_model = LinkTreeModel([jl1, jl2], g_acc=g)
    controller = Simple2RRWPController(jl2.q)
    return controller, plant_model, fict_model, jl1, jl2

def test_fictitious_link_sym():
    controller, plant_model, fict_model, jl1, jl2 = sym_models()
    H = fict_model.composite_inertia(dtype='object')
    print("H[0][1]", H[0][1])
    print("H[1][1]", H[1][1])
    Tc02 = -H[1][1] / H[0][1] / g
    print("Tc02", Tc02)
    D = H[1][2] * H[0][1] - H[1][1] * H[0][2]
    print(D)
    Y1 = H[0][1] / D
    Y2 = H[1][1] / D / g
    print("Y1", Y1)
    print("Y2", Y2)

def test_fictitious_link_plot():
    controller, plant_model, fict_model, jl1, jl2 = models()
    H = fict_model.composite_inertia()
    Tc0 = np.sqrt(-H[1][1] / H[0][1] / controller.g_acc)
    dt = 0.001
    ts, qcs = sample_input(dt)
    qfs = list(reversed(lpf(list(reversed(qcs)), 1.0*Tc0, dt)))

    import matplotlib.pyplot as plt
    plt.plot(ts, qcs, label="qc")
    plt.plot(ts, qfs, label="qf")
    q1 = []
    q2 = []
    for t,qc in zip(ts, qfs):
        q1.append(jl1.q)
        q2.append(jl2.q)
        tau2 = controller.tau(t, qc, dt, fict_model)
        jl2.tau = tau2
        plant_model.step(dt)
        fict_model.update()
    plt.plot(ts, q1, label="q1")
    plt.plot(ts, q2, label="q2")
    plt.legend()
    plt.show()


def test_fictitious_link_view():
    controller, plant_model, fict_model, jl1, jl2 = models()

    import graphic
    viewer = graphic.Viewer(scale=200, offset=[0, -0.1])
    t = 0
    tau2 = 0
    qc_ref = 0
    qc = 0
    step_flag = False
    def event_handler(key, shifted):
        nonlocal qc_ref, step_flag
        if key == 'q':
            sys.exit()
        elif key == 'k':
            qc_ref = qc_ref + 10*np.pi/180
        elif key == 'j':
            qc_ref = qc_ref - 10*np.pi/180
        elif key == 'space':
            step_flag = step_flag ^ True
    while True:
        viewer.handle_event(event_handler)
        viewer.clear()
        viewer.text([f"q1:{jl1.q:.03f} q2:{jl2.q:.03f}",
            f"t: {t:.03f}  qc:{qc:.03f} tau2:{tau2:.03f}"])
        viewer.draw(plant_model.drawCmds())
        viewer.flush()

        if not step_flag:
            continue

        H = fict_model.composite_inertia()
        Tc0 = np.sqrt(-H[1][1] / H[0][1] / controller.g_acc)
        qc = qc * 0.99 + qc_ref * 0.01
        #qc = qc_ref
        dt = 0.003
        tau2 = controller.tau(t, qc, dt, fict_model, 3*Tc0)
        jl2.tau = tau2
        plant_model.step(dt)
        fict_model.update()
        t = t + dt


if __name__ == '__main__':
    #test_fictitious_link_sym()
    test_fictitious_link_plot()
    #test_fictitious_link_view()

