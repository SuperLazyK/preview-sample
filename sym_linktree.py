from sympy import *
from sym_jointlink import *
import scipy
#from sympy import init_printing
#init_printing() 


# NOTE: the original index 1,2,... and 0 means base body(basically not moved)
# NOTE: body velocity in the body coordinate does not mean 0 (5.14)
class LinkTreeModel:

    def __init__(self, jointlinks, g, parent_idx=None):
        self.jointlinks = jointlinks
        self.NB = len(jointlinks)
        self.dim = 3 # num of dimension of spatial/planar vector
        self.X_parent_to = [zeros(self.dim, self.dim) for i in range(self.NB)]
        self.use_psudo_acc = False
        self.g = g
        if self.use_psudo_acc:
            self.accb = Matrix([0, 0, g]) # base link acc upper-ward
        else:
            self.accb = zeros(self.dim, 1)
        self.velb = zeros(self.dim, 1)

        self.X_r_to = [None for i in range(self.NB)]
        self.vel = [None for i in range(self.NB)]

        if parent_idx is None:
            self.parent_idx = list(range(-1, self.NB-1))
        else:
            for i in parent_idx:
                assert -1 <= i and i < self.NB
            self.parent_idx = parent_idx

        self.Ic = [zeros(self.dim, self.dim) for i in range(self.NB)]

        # symbolic calculation should be executed only once
        self.update_vel_X()


    def update_vel_X(self):
        for i in range(self.NB):
            I  = self.jointlinks[i].I
            XJ = self.jointlinks[i].XJ()
            j = self.parent(i)
            if j != -1: # parent is not root
                XT = self.jointlinks[j].XT
            else:
                XT = eye(self.dim)
            X_j_to_i = XJ * XT
            self.X_parent_to[i] = X_j_to_i

            if j != -1: # parent is not root
                self.X_r_to[i] = X_j_to_i * self.X_r_to[j]
                velj = self.vel[j]
            else:
                self.X_r_to[i] = X_j_to_i
                velj = self.velb

            vJ = self.jointlinks[i].vJ()
            self.vel[i] = X_j_to_i * velj + vJ

    def q(self):
        return Matrix([jl.q for jl in self.jointlinks])

    def dq(self):
        return Matrix([jl.dq for jl in self.jointlinks])

    def parent(self, i):
        return self.parent_idx[i]

    # recursive newton-euler on each body coordinate
    def inverse_dynamics(self, ddq, fext):
        q = self.q()
        dq = self.dq()
        NB = self.NB
        dim = self.dim

        assert NB == len(ddq)
        assert NB == len(fext)
        assert (dim, 1) == fext[0].shape

        acc = [zeros(dim, 1) for i in range(NB)]
        tau = [0 for i in range(NB)]
        f = [zeros(dim, 1) for i in range(NB)]

        S = [self.jointlinks[i].S() for i in range(NB)]

        for i in range(NB):
            I  = self.jointlinks[i].I
            vJ = self.jointlinks[i].vJ()
            cJ = self.jointlinks[i].cJ()
            j = self.parent(i)
            X_j_to_i = self.X_parent_to[i]

            if j != -1: # parent is not root
                accj = acc[j]
            else:
                accj = self.accb

            vel = self.vel[i]
            acc[i] = X_j_to_i * accj  + S[i] * ddq[i] + cJ + crm(vel) * vJ

            X_r_to = self.X_r_to[i]

            if self.use_psudo_acc:
                f[i] = I * acc[i] + crf(vel) * I * vel - dualm(X_r_to) * fext[i]
            else:
                accg_i = X_r_to * Matrix([0, 0, -self.g])
                f[i] = I * (acc[i] - accg_i) + crf(vel) * I * vel - dualm(X_r_to) * fext[i]

        for i in range(NB-1, -1, -1):
            tau[i] = S[i].dot(f[i])
            j = self.parent(i)
            if j != -1: # parent is root
                X_j_to_i = self.X_parent_to[i]
                f[j] = f[j] + X_j_to_i.T * f[i]

        return Matrix(tau)

    def composite_inertia(self):

        X_parent_to = self.X_parent_to
        NB = self.NB
        dim = self.dim
        H = zeros(NB, NB)
        S = [self.jointlinks[i].S() for i in range(NB)]

        Ic = [zeros(dim, dim) for i in range(NB)]
        for i in range(NB):
            Ic[i] = self.jointlinks[i].I

        for i in range(NB-1, -1, -1):
            j = self.parent(i)
            if j != -1: # parent is root
                X_j_to_i = X_parent_to[i]
                Ic[j] = Ic[j] + X_j_to_i.T * Ic[i] * X_j_to_i
            F = Ic[i] * S[i]
            H[i,i] = S[i].T * F

            j = i
            while self.parent(j) != -1:
                F = X_parent_to[j].T * F
                j = self.parent(j)
                H[i,j] = F.T * S[j]
                H[j,i] = H[i,j]

        # local to global
        for i in range(NB):
            self.Ic[i] = self.X_r_to[i].T * Ic[i] * self.X_r_to[i]

        return H

    # symbolic calculation should be executed only once
    def calc_H_C_tau(self, fext=None):
        if fext is None:
            fext = [zeros(self.dim, 1) for i in range(self.NB)]
        tau = Matrix([jl.force() for jl in self.jointlinks])
        C = self.inverse_dynamics([0 for i in range(self.NB)], fext)
        H = self.composite_inertia()
        return H, C, tau

    # q H q is also ok
    def kinetic_energy(self):
        T = 0
        for i in range(self.NB):
            T = T + (self.jointlinks[i].vJ().T * self.Ic[i] * self.jointlinks[i].vJ())[0,0]
        return T

    def potential_energy(self):
        U = 0
        for i in range(self.NB):
            X_r_to_i = self.X_r_to[i]
            _, _, _, y = Xtoscxy(self.jointlinks[i].Xc * X_r_to_i)
            U = U + y * self.g * self.jointlinks[i].m
        return U

    def syn_inertia_info(self):
        I = self.Ic[0][0,0]
        m = self.Ic[0][1,1]
        cog = Matrix([self.Ic[0][0,2]/m, -self.Ic[0][0,1]/m])
        return I, m, cog

    # foward dynqmics
    def gen_ddq_f(self, q_sym_list, dq_sym_list, other_sym_list=[], ctx={}, fext=None):
        H, C, tau = self.calc_H_C_tau(fext)
        syms = q_sym_list +  dq_sym_list + other_sym_list
        Hevalf = lambdify(syms, H.subs(ctx))
        rhs = lambdify(syms, (tau - C).subs(ctx))
        def ddq_f(qv, dqv, uv):
            b = rhs(*qv, *dqv, *uv).reshape(-1)
            A = Hevalf(*qv, *dqv, *uv)
            return np.linalg.solve(A, b)
        return ddq_f


    def gen_draw_cmds(self, q_sym_list, dq_sym_list, ctx):
        q_dq = q_sym_list +  dq_sym_list
        X_r_to_fs = [lambdify(q_dq, self.X_r_to[i].subs(ctx)) for i in range(self.NB)]

        def draw_cmds(qv, dqv):
            cmds = []
            for i in range(self.NB):
                X = X_r_to_fs[i](*qv, *dqv)
                cmds = cmds + self.jointlinks[i].drawCmd(X, ctx)
            return cmds
        return draw_cmds



#def ref(q, dq, I1, I2):
#    m1 = 1
#    m2 = 1
#    l1 = 2
#    l2 = 1
#    q1 = q[0]
#    q2 = q[1]
#    dq1 = dq[0]
#    dq2 = dq[1]
#    g = 9.8
#    H = np.zeros((2,2))
#    H[0,0] = (4*l1*l2*m2*cos(q2)+(l2**2+4*l1**2)*m2+l1**2*m1+4*I2+4*I1)/4.0E+0
#    H[0,1] = (2*l1*l2*m2*cos(q2)+l2**2*m2+4*I2)/4.0E+0
#    H[1,0] = (2*l1*l2*m2*cos(q2)+l2**2*m2+4*I2)/4.0E+0
#    H[1,1] = (l2**2*m2+4*I2)/4.0E+0
#
#    C = np.zeros(2)
#    C[0] = (-(dq2**2*l1*l2*m2*cos(q1)*sin(q2+q1))/2.0E+0)-dq1*dq2*l1*l2*m2*cos(q1)*sin(q2+q1)+(dq2**2*l1*l2*m2*sin(q1)*cos(q2+q1))/2.0E+0+dq1*dq2*l1*l2*m2*sin(q1)*cos(q2+q1)+(g*l2*m2*cos(q2+q1))/2.0E+0+g*l1*m2*cos(q1)+(g*l1*m1*cos(q1))/2.0E+0
#    C[1] = (dq1**2*l1*l2*m2*cos(q1)*sin(q2+q1))/2.0E+0-(dq1**2*l1*l2*m2*sin(q1)*cos(q2+q1))/2.0E+0+(g*l2*m2*cos(q2+q1))/2.0E+0
#
#    return np.linalg.solve(H, -C)
#
#

g = symbols('g')
r,l,th,phi,Iw,Ib,mw,mb = symbols('r l th phi Iw Ib mw mb')
dth,dphi = symbols('dth dphi')
ddth,ddphi = symbols('ddth ddphi')
q1, dq1, l1, m1 = symbols('q1 dq1 l1 m1')
q2, dq2, l2, m2 = symbols('q2 dq2 l2 m2')
q3, dq3, l3, m3 = symbols('q3 dq3 l3 m3')
q_sym = [q1, q2, q3]
dq_sym = [dq1, dq2, dq3]
context = { l1: 0.2, l2: 0.25, l3: 0.35,
        m1: 0.7, m2: 0.5, m3: 0.3,
        g: 9.81
        }


def models(use3=True, fict=False):

    jl0 = StickJointLink(0, 0, PrismaticJoint(), 0, 0)
    jl1 = StickJointLink(m1, l1, RevoluteJoint(), q1, dq1, cx=l1, Icog=0)
    jl2 = StickJointLink(m2, l2, RevoluteJoint(), q2, dq2, cx=l2, Icog=0)
    jl3 = StickJointLink(m3, l3, RevoluteJoint(), q3, dq3, cx=l3, Icog=0)
    if use3:
        if fict:
            model = LinkTreeModel([jl0, jl1, jl2, jl3], g)
        else:
            model = LinkTreeModel([jl1, jl2, jl3], g)
    else:
        model = LinkTreeModel([jl1, jl2], g)

    return model


#def test_2link():
#    model, gen_state, g = models(False)
#    H, C, tau = model.calc_H_C_tau()
#    Hmac = zeros(2,2)
#    Cmac = zeros(2)
#    Hmac[0,0] = 2*l1*l2*m2*cos(q2)+(l2**2+l1**2)*m2+l1**2*m1
#    Hmac[0,1] = l1*l2*m2*cos(q2)+l2**2*m2
#    Hmac[1,0] = l1*l2*m2*cos(q2)+l2**2*m2
#    Hmac[1,1] = l2**2*m2
#    Cmac[0,0] = (-dq2**2*l1*l2*m2*cos(q1)*sin(q2+q1))-2*dq1*dq2*l1*l2*m2*cos(q1)*sin(q2+q1)+dq2**2*l1*l2*m2*sin(q1)*cos(q2+q1)+2*dq1*dq2*l1*l2*m2*sin(q1)*cos(q2+q1)+g*l2*m2*cos(q2+q1)+g*l1*m2*cos(q1)+g*l1*m1*cos(q1) 
#    Cmac[1,0] = dq1**2*l1*l2*m2*cos(q1)*sin(q2+q1)-dq1**2*l1*l2*m2*sin(q1)*cos(q2+q1)+g*l2*m2*cos(q2+q1)                                                                                                                
#
#    for i in range(2):
#        for j in range(2):
#            print("H",i,j,":",simplify(H[i,j]-Hmac[i,j]))
#    for i in range(2):
#        print("C",i,":",simplify(C[i]-Cmac[i,0]))


#def test_3link():
#    model, gen_state, g = models()
#    H, C, tau = model.calc_H_C_tau()
#    Hmac = zeros(3,3)
#    Cmac = zeros(3)
#    Hmac[0,0] = (-2*l1*l3*m3*sin(q2)*sin(q3))+(2*l1*l3*m3*cos(q2)+2*l2*l3*m3)*cos(q3)+(2*l1*l2*m3+2*l1*l2*m2)*cos(q2)+(l3**2+l2**2+l1**2)*m3+(l2**2+l1**2)*m2+l1**2*m1
#    Hmac[0,1] = (-l1*l3*m3*sin(q2)*sin(q3))+(l1*l3*m3*cos(q2)+2*l2*l3*m3)*cos(q3)+(l1*l2*m3+l1*l2*m2)*cos(q2)+(l3**2+l2**2)*m3+l2**2*m2
#    Hmac[0,2] = (-l1*l3*m3*sin(q2)*sin(q3))+(l1*l3*m3*cos(q2)+l2*l3*m3)*cos(q3)+l3**2*m3
#    Hmac[1,0] = (-l1*l3*m3*sin(q2)*sin(q3))+(l1*l3*m3*cos(q2)+2*l2*l3*m3)*cos(q3)+(l1*l2*m3+l1*l2*m2)*cos(q2)+(l3**2+l2**2)*m3+l2**2*m2
#    Hmac[1,1] = 2*l2*l3*m3*cos(q3)+(l3**2+l2**2)*m3+l2**2*m2
#    Hmac[1,2] = l2*l3*m3*cos(q3)+l3**2*m3
#    Hmac[2,0] = (-l1*l3*m3*sin(q2)*sin(q3))+(l1*l3*m3*cos(q2)+l2*l3*m3)*cos(q3)+l3**2*m3
#    Hmac[2,1] = l2*l3*m3*cos(q3)+l3**2*m3
#    Hmac[2,2] = l3**2*m3
#    Cmac[0,0] = (-dq3**2*l2*l3*m3*cos(q2+q1)*sin(q3+q2+q1))-2*dq2*dq3*l2*l3*m3*cos(q2+q1)*sin(q3+q2+q1)-2*dq1*dq3*l2*l3*m3*cos(q2+q1)*sin(q3+q2+q1)-dq3**2*l1*l3*m3*cos(q1)*sin(q3+q2+q1)-2*dq2*dq3*l1*l3*m3*cos(q1)*sin(q3+q2+q1)-2*dq1*dq3*l1*l3*m3*cos(q1)*sin(q3+q2+q1)-dq2**2*l1*l3*m3*cos(q1)*sin(q3+q2+q1)-2*dq1*dq2*l1*l3*m3*cos(q1)*sin(q3+q2+q1)+dq3**2*l2*l3*m3*sin(q2+q1)*cos(q3+q2+q1)+2*dq2*dq3*l2*l3*m3*sin(q2+q1)*cos(q3+q2+q1)+2*dq1*dq3*l2*l3*m3*sin(q2+q1)*cos(q3+q2+q1)+dq3**2*l1*l3*m3*sin(q1)*cos(q3+q2+q1)+2*dq2*dq3*l1*l3*m3*sin(q1)*cos(q3+q2+q1)+2*dq1*dq3*l1*l3*m3*sin(q1)*cos(q3+q2+q1)+dq2**2*l1*l3*m3*sin(q1)*cos(q3+q2+q1)+2*dq1*dq2*l1*l3*m3*sin(q1)*cos(q3+q2+q1)+g*l3*m3*cos(q3+q2+q1)-dq2**2*l1*l2*m3*cos(q1)*sin(q2+q1)-2*dq1*dq2*l1*l2*m3*cos(q1)*sin(q2+q1)-dq2**2*l1*l2*m2*cos(q1)*sin(q2+q1)-2*dq1*dq2*l1*l2*m2*cos(q1)*sin(q2+q1)+dq2**2*l1*l2*m3*sin(q1)*cos(q2+q1)+2*dq1*dq2*l1*l2*m3*sin(q1)*cos(q2+q1)+dq2**2*l1*l2*m2*sin(q1)*cos(q2+q1)+2*dq1*dq2*l1*l2*m2*sin(q1)*cos(q2+q1)+g*l2*m3*cos(q2+q1)+g*l2*m2*cos(q2+q1)+g*l1*m3*cos(q1)+g*l1*m2*cos(q1)+g*l1*m1*cos(q1)
#    Cmac[1,0] = (-dq3**2*l2*l3*m3*cos(q2+q1)*sin(q3+q2+q1))-2*dq2*dq3*l2*l3*m3*cos(q2+q1)*sin(q3+q2+q1)-2*dq1*dq3*l2*l3*m3*cos(q2+q1)*sin(q3+q2+q1)+dq1**2*l1*l3*m3*cos(q1)*sin(q3+q2+q1)+dq3**2*l2*l3*m3*sin(q2+q1)*cos(q3+q2+q1)+2*dq2*dq3*l2*l3*m3*sin(q2+q1)*cos(q3+q2+q1)+2*dq1*dq3*l2*l3*m3*sin(q2+q1)*cos(q3+q2+q1)-dq1**2*l1*l3*m3*sin(q1)*cos(q3+q2+q1)+g*l3*m3*cos(q3+q2+q1)+dq1**2*l1*l2*m3*cos(q1)*sin(q2+q1)+dq1**2*l1*l2*m2*cos(q1)*sin(q2+q1)-dq1**2*l1*l2*m3*sin(q1)*cos(q2+q1)-dq1**2*l1*l2*m2*sin(q1)*cos(q2+q1)+g*l2*m3*cos(q2+q1)+g*l2*m2*cos(q2+q1)
#    Cmac[2,0] = dq2**2*l2*l3*m3*cos(q2+q1)*sin(q3+q2+q1)+2*dq1*dq2*l2*l3*m3*cos(q2+q1)*sin(q3+q2+q1)+dq1**2*l2*l3*m3*cos(q2+q1)*sin(q3+q2+q1)+dq1**2*l1*l3*m3*cos(q1)*sin(q3+q2+q1)-dq2**2*l2*l3*m3*sin(q2+q1)*cos(q3+q2+q1)-2*dq1*dq2*l2*l3*m3*sin(q2+q1)*cos(q3+q2+q1)-dq1**2*l2*l3*m3*sin(q2+q1)*cos(q3+q2+q1)-dq1**2*l1*l3*m3*sin(q1)*cos(q3+q2+q1)+g*l3*m3*cos(q3+q2+q1)
#
#    #print(model.kinetic_energy())
#    #Kmac = (m3*(((-(dq3+dq2+dq1)*l3*sin(q3+q2+q1))-(dq2+dq1)*l2*sin(q2+q1)-dq1*l1*sin(q1))**2+((dq3+dq2+dq1)*l3*cos(q3+q2+q1)+(dq2+dq1)*l2*cos(q2+q1)+dq1*l1*cos(q1))**2))/2.0E+0+(m2*(((-(dq2+dq1)*l2*sin(q2+q1))-dq1*l1*sin(q1))**2+((dq2+dq1)*l2*cos(q2+q1)+dq1*l1*cos(q1))**2))/2.0E+0+(m1*(dq1**2*l1**2*sin(q1)**2+dq1**2*l1**2*cos(q1)**2))/2.0E+0
#    #K = (Matrix([dq1,dq2,dq3]).T * Hmac * Matrix([dq1,dq2,dq3]))[0,0]/2
#    #Ksimp = simplify(K)
#    #Kmacsimp = simplify(Kmac)
#    #print(Ksimp)
#    #print(Kmacsimp)
#    #print(2*(Kmacsimp-Ksimp))
#    #for i in range(3):
#    #    for j in range(3):
#    #        print("H",i,j,":",simplify(H[i,j]-Hmac[i,j]))
#    for i in range(3):
#        print("C",i,":",simplify(C[i]-Cmac[i,0]))
#
##def test_4link():
##    model, gen_state, g = models(fict=True)
##    H, C, tau = model.calc_H_C_tau()
##    for i in range(4):
##        for j in range(4):
##            print("H",i,j,":",simplify(H[i,j]))


def test_2link_view():
    model = models(use3 = False)
    q = np.array([np.pi/2+0.01, 0, 0], dtype=np.float64)
    dq = np.array([0, 0, 0], dtype=np.float64)
    ddq_f = model.gen_ddq_f(q_sym, dq_sym, context)
    draw_cmds = model.gen_draw_cmds(q_sym, dq_sym, context)

    import graphic
    viewer = graphic.Viewer(scale=200, offset=[0, -0.1])
    dt = 0.001
    t = 0
    while True:
        t = t + dt
        ddq = ddq_f(q, dq)

        dq[:2] = dq[:2] + ddq * dt
        q = q + dq * dt

        #scipy.integrate.RK45(ddq, t, q, dq)
        k = model.kinetic_energy()
        cmds = draw_cmds(q, dq)
        viewer.clear()
        viewer.handle_event(graphic.default_event_handler)
        viewer.text([f"t: {t:.03f} :q {q[0]:.03f} {q[1]:.03f} {q[2]:.03f}"])
        viewer.draw(cmds)
        viewer.flush(dt)


if __name__ == '__main__':
    #test_2link()
    test_2link_view()
    #test_4link()

