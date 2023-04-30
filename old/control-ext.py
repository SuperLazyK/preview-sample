from linktree import *
import numpy as np
from jointlink import *
import sys
import time
from arith import *
from signal_proc import *

# NOTE:
# q0 is fict
# q1 is passive
# q2 is active for balance
# q3 is active for attitude
class SimpleRPController():
    def __init__(self, qc, g_acc = 9.81):
        self.g_acc = g_acc
        self.Ls = [0, 0, 0]
        y2c = qc[0]
        y3c = np.array(qc[1:])
        self.y2cs = []
        self.y3cs = []
        self.y2fs = []
        self.y3fs = []
        self.ts = [None, None, None,]

    def tau(self, t, qc, dt, fict_model, previewT = 0):
        H = fict_model.composite_inertia()
        jl1 = fict_model.jointlinks[1]
        jl2 = fict_model.jointlinks[2]
        jl3s = fict_model.jointlinks[3:]
        if H[1][1] / H[0][1] >= 0:
            print("ERROR: Tc**2 <= 0")
            return None
        assert H[1][1] / H[0][1] < 0
        Tc = np.sqrt(-H[1][1] / H[0][1] / self.g_acc)

        # observe
        dq1 = jl1.dq
        dy2 = jl2.dq
        dy3 = np.array([j.dq for j in jl3s])
        y3 = np.array([j.q for j in jl3s])
        L = H[1][1] * dq1 + H[1][2] * dy2 + H[1][3:] @ dy3
        print("t:", t)
        print("q1", jl1.q)
        print("yc2", qc[0])
        print("q2", jl2.q)
        print("yc3", qc[1:])
        print("q3", y3)
        print("L", L)
        self.ts.append(t)
        self.Ls.append(L)

        # control
        Ls = self.Ls
        if previewT >= dt:
            y2c = qc[0]
            y3c = np.array(qc[1:])
            N = int(previewT / dt)
            while len(self.y2cs) < N:
                self.y2cs.append(qc[0])
                self.y3cs.append(np.array(qc[1:]))
            self.y2cs.append(y2c)
            self.y3cs.append(y3c)
            y2f = rlpf(self.y2cs[-(N+1):], Tc, dt)
            y3f = rlpf(self.y3cs[-(N+1):], Tc, dt)
            while len(self.y2fs) < 2:
                self.y2fs.append(y2f)
                self.y3fs.append(y3f)
            self.y2fs.append(y2f)
            self.y3fs.append(y3f)
            y2cs = self.y2fs
            y3cs = self.y3fs
        else:
            y2c = qc[0]
            y3c = np.array(qc[1:])
            if len(self.y2cs) == 0:
                self.y2cs.append(y2c)
                self.y2cs.append(y2c)
                self.y3cs.append(y3c)
                self.y3cs.append(y3c)
            self.y2cs.append(y2c)
            self.y3cs.append(y3c)
            y2cs = self.y2cs
            y3cs = self.y3cs

        L = Ls[-1]
        dL = diff(Ls, 1, dt)
        ddL = diff(Ls, 2, dt)
        y2c = y2cs[-1]
        y3c = np.array(y3cs[-1])
        dy3c = diff(y3cs , 1, dt)
        ddy3c = diff(y3cs, 2, dt)
        ddy3 = ddy3c + 400 * (y3c - y3) + 40 * (dy3c - dy3)

        D = H[1][2] * H[0][1] - H[1][1] * H[0][2]
        if D == 0:
            print("ERROR: D==0")
            return None
        assert D != 0

        E = H[1][3:]*H[0][1] - H[1][1] * H[0][3:]
        Y1 = H[0][1] / D
        Y2 = H[1][1] / D / self.g_acc
        I,m,cog = fict_model.syn_inertia_info()
        print("I, m, cog", I, m, cog)
        print("Tc", Tc)
        print("Gv2", -(H[1][2] * H[0][1] - H[1][1] * H[0][2])/(m*H[1,1]))
        print("Gv3", -(H[1][3] * H[0][1] - H[1][1] * H[0][3])/(m*H[1,1]))
        print("Y1", Y1)
        print("Y2", Y2)
        #print("dy3", dy3)
        #print("ddy3", ddy3)
        Y3 = E / D
        y2 = jl2.q
        qc = y2c
        dqc = diff(y2cs, 1, dt)
        ddqc = diff(y2cs, 2, dt)
        print("y2cs: ", y2cs[-3:])

        H2 = np.zeros_like(H)
        H2[2,0] = -1
        H2[3:,1:-2] = -1
        H2[:,-2] = H[:,1]
        H2[:,-1] = H[:,2]
        assert np.linalg.matrix_rank(H2) == H.shape[0]

        C = fict_model.inverse_dynamics(np.zeros(fict_model.NB), np.zeros((fict_model.NB, fict_model.dim)))
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

        #alpha1 = 0
        #alpha2 = 0
        alpha2 = 1 / (lmd3 * lmd4)
        alpha1 = -alpha2 * (lmd3 + lmd4)
        #print(alpha1, alpha2)
        u = qc + alpha1 * dqc + alpha2 * ddqc
        print(f"u: {u:.03f} = {qc:.03f} + {alpha1:.03f} * {dqc:.03f} + {alpha2:.03f} * {ddqc:.03f}")

        dddL =  kdd * ddL + kd * dL + kL * (L - Y3 @ dy3 / Y1) + kq * (y2 - u)
        print(f"dddL: {dddL:.01f} = {kdd:.01f} * {ddL:.01f} + {kd:.01f} * {dL:.01f} + {kL:.01f} * {L:.01f} + {kq:.01f} * ({y2:.01f} - {u:.01f})")
        #print("detH2", np.linalg.det(H2))
        offset = np.zeros_like(C)
        offset[0] = dddL /self.g_acc + H[0,3:] @ ddy3
        offset[1] = H[1, 3:] @ ddy3
        offset[2] = H[2, 3:] @ ddy3
        offset[3] = H[3, 3:] @ ddy3
        tau = np.linalg.solve(H2, -C - offset)[:-2]
        print("tau", tau)
        return tau

def models():
    g_acc=9.81
    jl0 = StickJointLink(0, 0, PrismaticJoint())
    jl1 = StickJointLink(0.7, 0.2, RevoluteJoint(), cx = 0.2, Icog=0, q=np.pi/2)
    jl2 = StickJointLink(0.5, 0.25, RevoluteJoint(), cx = 0.25, Icog=0)
    jl3 = StickJointLink(0.3, 0.35, RevoluteJoint(), cx = 0.35, Icog=0)
    fict_model = LinkTreeModel([jl0, jl1, jl2, jl3], g_acc=g_acc)
    plant_model = LinkTreeModel([jl1, jl2, jl3], g_acc=g_acc)
    qc = [jl2.q, jl3.q]
    controller = SimpleRPController(qc, g_acc = g_acc)
    return controller, plant_model, fict_model, jl1, jl2, jl3


def sample_input(dt = 0.001):
    y2cpfs = [ (lambda t:0, [-2,1])
             , (lambda t:1.5, [1,2.75])
             , (lambda t:1.5-3*(t-2.75), [2.75,3.25])
             , (lambda t:0, [3.25,4])
             , (lambda t:sin(np.pi*(t-4)), [4, 7])
             , (lambda t:0, [7, 11])
             , (lambda t:1.5*(t-11), [11, 12])
             , (lambda t:1.5, [12, 13])
             ]
    y3cpfs = [ (lambda t:0, [-2, 8] )
             , (lambda t:1.5*(t-8), [8, 9] )
             , (lambda t:1.5, [9, 13] )
             ]
    t = -2
    ts = []
    y2cs = []
    y3cs = []
    while t <13:
        y2c = sample(y2cpfs, t)
        y3c = sample(y3cpfs, t)
        y2cs.append(y2c)
        y3cs.append(y3c)
        ts.append(t)
        t = t + dt

    return ts, y2cs, y3cs


def test_fictitious_link_plot():
    g_acc=9.81
    controller, plant_model, fict_model, jl1, jl2, jl3 = models()
    H = fict_model.composite_inertia()
    Tc0 = np.sqrt(-H[1][1] / H[0][1] / g_acc)
    dt = 0.003
    ts, y2cs, y3cs = sample_input(dt)
    y2fs = list(reversed(lpf(list(reversed(y2cs)), 1.0*Tc0, dt)))
    y3fs = list(reversed(lpf(list(reversed(y3cs)), 1.0*Tc0, dt)))

    import matplotlib.pyplot as plt
    #plt.plot(ts, y3cs, label="y3c")
    #plt.plot(ts, y2cs, label="y2c")
    #plt.plot(ts, y2fs, label="y2f")
    #plt.plot(ts, y3fs, label="y3f")

    q1 = []
    q2 = []
    q3 = []
    q2c = []
    q3c = []
    tau2 = []
    tau3 = []
    tidx = []

    for t,y2f,y3f in zip(ts, y2fs, y3fs):
        print(f"CHECK t:{t:.03f}, q1:{jl1.q*180/np.pi:.01f}, q2:{jl2.q*180/np.pi:.01f}, q3:{jl3.q*180/np.pi:.01f}")
        qc = [y2f, y3f]
        print(f"CHECK-c q2c:{y2f*180/np.pi:.01f}, q3c:{y3f*180/np.pi:.01f}")
        tau = controller.tau(t, qc, dt, fict_model)
        if tau is None:
            break
        jl2.tau = tau[0]
        jl3.tau = tau[1]
        plant_model.step(dt)
        fict_model.update()
        if tau[0] > 100 or tau[1] > 100:
            print(jl1.q - np.pi/2)
            print(jl2.q)
            print(jl3.q)
            break
        if t >= 0:
            tidx.append(t)
            q2c.append(y2f)
            q3c.append(y3f)
            q1.append(jl1.q - np.pi/2)
            q2.append(jl2.q)
            q3.append(jl3.q)
            tau2.append(tau[0])
            tau3.append(tau[1])

    plt.plot(tidx, q1, label="q1")
    plt.plot(tidx, q2, label="q2")
    plt.plot(tidx, q3, label="q3")
    plt.plot(tidx, q2c, label="q2c")
    plt.plot(tidx, q3c, label="q3c")
    plt.plot(tidx, tau2, label="tau2")
    plt.plot(tidx, tau3, label="tau3")
    plt.legend()
    plt.show()


def test_fictitious_link_view():
    controller, plant_model, fict_model, jl1, jl2, jl3 = models()

    import graphic
    viewer = graphic.Viewer(scale=200, offset=[0, -0.1])
    t = 0
    tau2 = 0
    tau3 = 0
    y2c_ref = 0
    y3c_ref = 0
    y2c = 0
    y3c = 0
    step_flag = False
    def event_handler(key, shifted):
        nonlocal y2c_ref, y3c_ref, step_flag
        if key == 'q':
            sys.exit()
        elif key == 'k':
            y2c_ref = y2c_ref + 10*np.pi/180
        elif key == 'j':
            y2c_ref = y2c_ref - 10*np.pi/180
        elif key == 'h':
            y3c_ref = y3c_ref + 10*np.pi/180
        elif key == 'l':
            y3c_ref = y3c_ref - 10*np.pi/180
        elif key == 'space':
            step_flag = step_flag ^ True
    while True:
        viewer.handle_event(event_handler)
        viewer.clear()
        viewer.text([f"t: {t:.03f} q1:{jl1.q:.03f}",
            f"q2:{jl2.q:.03f} y2c:{y2c:.03f} tau2:{jl2.tau:.03f}",
            f"q3:{jl3.q:.03f} y3c:{y3c:.03f} tau3:{jl3.tau:.03f}"])
        viewer.draw(plant_model.drawCmds())
        viewer.flush()

        if not step_flag:
            continue

        y2c = y2c * 0.99 + y2c_ref * 0.01
        y3c = y3c * 0.99 + y3c_ref * 0.01
        dt = 0.003
        H = fict_model.composite_inertia()
        Tc0 = np.sqrt(-H[1][1] / H[0][1] / controller.g_acc)
        tau = controller.tau(t, [y2c, y3c], dt, fict_model, 3*Tc0)
        jl2.tau = tau[0]
        jl3.tau = tau[1]
        plant_model.step(dt)
        fict_model.update()
        t = t + dt

import sympy
l1, l2, l3, m1, m2, m3, g = sympy.symbols('l1 l2 l3 m1 m2 m3 g')

def sym_models():
    jl0 = StickJointLink(0, 0, PrismaticJoint())
    jl1 = StickJointLink(m1, l1, RevoluteJoint(), Icog=0, q=np.pi/2)
    jl2 = StickJointLink(m2, l2, RevoluteJoint(), Icog=0)
    jl3 = StickJointLink(m3, l3, RevoluteJoint(), Icog=0)
    fict_model = LinkTreeModel([jl0, jl1, jl2, jl3], g_acc=g)
    plant_model = LinkTreeModel([jl1, jl2, jl3], g_acc=g)
    controller = SimpleRPController([jl2.q, jl3.q])
    return controller, plant_model, fict_model, jl1, jl2, jl3

def test_fictitious_link_sym():
    controller, plant_model, fict_model, jl1, jl2, jl3 = sym_models()
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


if __name__ == '__main__':
    #test_fictitious_link_sym()
    test_fictitious_link_plot()
    #test_fictitious_link_view()
