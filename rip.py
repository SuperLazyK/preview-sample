from sym_linktree import *
import numpy as np
from sym_jointlink import *
import sys
import time
from sym_arith import *
from sympy import *
import control as ct

# ref
# https://www.ascento.ethz.ch/wp-content/uploads/2019/05/AscentoPaperICRA2019.pdf
# https://arxiv.org/pdf/2005.11431.pdf
# https://www2.akita-nct.ac.jp/libra/report/46/46038.pdf

r,l,Iw,Ib,mw,mb = symbols('r l Iw Ib mw mb')
q1, dq1, ddq1 = symbols('q1 dq1 ddq1')
q2, dq2, ddq2 = symbols('q2 dq2 ddq2')
th, dth, ddth  = symbols('th dth ddth')
phi,dphi,ddphi = symbols('phi dphi ddphi')
u = symbols('u')
q_sym = [q1, q2]
dq_sym = [dq1, dq2]
u_sym =[u]


def sym_models():
    # initial wheel angle should be vertical
    jl1 = WheelJointLink(mw, r, RackPinionJoint(r), q1+pi/2, dq1, Icog=Iw)
    jl2 = StickJointLink(mb, l, RevoluteJoint(), q2, dq2, cx=l, Icog=Ib, tau=u)
    plant_model = LinkTreeModel([jl1, jl2], g)
    return plant_model, jl1, jl2


class RIPController():
    def __init__(self, model, context, reuse=True):
        self.model = model
        self.context = context
        x = Matrix([dphi, dth, th])
        self.u = u
        self.x = x

        if reuse:
            self.A = np.array([[0, 0, 129.982499999994], [0, 0, 17.1674999999997], [0, 1, 0]])
            self.B = np.array([[-60.3888888888863], [-2.94444444444431], [0]])
            return

        lhs = ((H0) * Matrix([[ddq1],[ddq2]]) + C0 - Matrix([[0], [u]])).subs(context)

        # phi:-q2
        # th:q2+q1
        # q1 = phi + th
        # q2 = -phi
        lhs1 = simplify(lhs.subs([ (q1,phi+th)
                        , (q2, -phi)
                        , (dq1, dphi+dth)
                        , (dq2, -dphi)
                        , (ddq1, ddphi+ddth)
                        , (ddq2, -ddphi)
                        ]))
        lhs2 = Matrix([[lhs1[0]], [lhs1[0] - lhs1[1]]])

        # liniarize
        lhs2 = lhs2.subs([ (cos(th), 1)
                         , (sin(th), th)
                         , (dth**2, 0)
                         ])

        H2 = lhs2.jacobian(Matrix([ddphi, ddth]))
        C2 = (lhs2 - H2 * Matrix([[ddphi], [ddth]]))

        H3 = H2.col_insert(2, Matrix([[0], [0]])).row_insert(2, Matrix([[0, 0, 1]]))
        rhs3 = H3.inv() * (-C2).row_insert(2, Matrix([[dth]]))

        A = (rhs3.jacobian(x))
        B = ((rhs3 - A * x).jacobian(Matrix([u])))
        D = (rhs3 - A * x - B * u)

        # H x' = Ax + Bu
        self.A = A
        self.B = B
        print(A)
        print(B)
        print(D)

    def test(self):
        ddq_f = self.model.gen_ddq_f(q_sym, dq_sym, u_sym, self.context)
        def nonlin_rhs(t, x, u, params):
            #x = Matrix([dphi, dth, phi, th])
            v_dphi = x[0]
            v_dth = x[1]
            v_phi = x[2]
            v_th = x[3]
            v_q = [v_phi + v_th, -v_phi]
            v_dq = [v_dphi + v_dth, -v_dphi]
            v_ddq = ddq_f(v_q, v_dq, [u[0]])
            return [-v_ddq[1], v_ddq[0] + v_ddq[1], v_dphi, v_dth]

        nonlin_cplant = ct.NonlinearIOSystem( nonlin_rhs, lambda t, x, u, params: [x[0], x[1],x[3]], inputs='u', outputs=('dphi', 'dth', 'th'), states=4, name='P')
        lin_cplant = ct.ss(self.A, self.B, np.identity(3), 0, name='LP', inputs=('u'), outputs=('dphi', 'dth', 'th'))

        Q = np.diag([1,1,1])
        R = 1
        K, _, E = ct.lqr(self.A, self.B, Q, R)
        #print("pole", E)
        #lin_csys = ct.ss(self.A - self.B @ K, self.B @ K, np.identity(3), 0)
        lin_csys = ct.feedback(ct.series(K, lin_cplant), np.identity(3), sign= -1)
        nonlin_csys = ct.feedback(ct.series(K, nonlin_cplant), np.identity(3), sign= -1)
        ts = np.linspace(0, 5, 100)
        #ts, xs = ct.step_response(lin_csys, input=0, T=T)
        #ts, xs = ct.input_output_response(lin_csys, ts, U=[1, 0, 0])
        ts, xs = ct.input_output_response(nonlin_csys, ts, U=[1, 0, 0])

        import matplotlib.pyplot as plt
        plt.subplot(111)
        plt.title("Identity weights")
        plt.plot(ts.T, xs[0].T, '-', label="phi'")
        plt.plot(ts.T, xs[1].T, '--', label="th'")
        plt.plot(ts.T, xs[2].T, '--', label="th")
        plt.legend()
        plt.show()

        pass

    def tau(self, ref_v, q, qd):
        return 0

def test():
    model, jl1, jl2 = sym_models()
    context = { l: 0.5, r: 0.05,
            mw: 1, mb: 9,
            Iw: 1./800, Ib: 2.25,
            g: 9.81
            }
    q = np.array([0, 0], dtype=np.float64)
    dq = np.array([0, 0], dtype=np.float64)
    controller = RIPController(model, context)

    controller.test()
    return

    ddq_f = model.gen_ddq_f(q_sym, dq_sym, u_sym, context)
    draw_cmds = model.gen_draw_cmds(q_sym, dq_sym, context)

    import graphic
    viewer = graphic.Viewer(scale=400, offset=[0, -0.1])
    dt = 0.001
    t = 0
    v_u = 0

    def event_handler(key, shifted):
        nonlocal v_u
        if key == 'q':
            sys.exit()
        elif key == 'k':
            v_u = 0.01
        elif key == 'j':
            v_u = -0.01

    while True:
        t = t + dt
        ddq = ddq_f(q, dq, [v_u])

        dq[:2] = dq[:2] + ddq * dt
        q = q + dq * dt

        #scipy.integrate.RK45(ddq, t, q, dq)
        k = model.kinetic_energy()
        cmds = draw_cmds(q, dq)
        viewer.handle_event(event_handler)
        viewer.clear()
        viewer.text([f"t: {t:.03f} :q {q[0]:.03f} {q[1]:.03f}"])
        viewer.draw(cmds)
        viewer.flush(dt)

if __name__ == '__main__':
    test()

