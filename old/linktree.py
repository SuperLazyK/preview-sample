import numpy as np
from numpy import sin, cos, abs
from jointlink import *

#np.set_printoptions(linewidth=np.inf)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=3)



# NOTE: the original index 1,2,... and 0 means base body(basically not moved)
# NOTE: body velocity in the body coordinate does not mean 0 (5.14)
class LinkTreeModel:

    def __init__(self, jointlinks, parent_idx=None, g_acc=9.8):
        self.jointlinks = jointlinks
        self.NB = len(jointlinks)
        self.dim = 3 # num of dimension of spatial/planar vector
        self.X_parent_to = [np.zeros((self.dim, self.dim)) for i in range(self.NB)]
        self.vel = [np.zeros(self.dim) for i in range(self.NB)] # vel of body i in body i coordinate
        self.use_psudo_acc = False
        self.g_acc = g_acc
        if self.use_psudo_acc:
            self.accb = np.array([0, 0, g_acc]) # base link acc upper-ward
        else:
            self.accb = np.zeros(self.dim)
        self.velb = np.zeros(self.dim)

        if parent_idx is None:
            self.parent_idx = list(range(-1, self.NB-1))
        else:
            for i in parent_idx:
                assert -1 <= i and i < self.NB
            self.parent_idx = parent_idx

        self.Ic = [np.zeros([self.dim, self.dim]) for i in range(self.NB)]

        self.update()

    def q(self):
        return np.array([jl.q for jl in self.jointlinks])

    def dq(self):
        return np.array([jl.dq for jl in self.jointlinks])

    def set_q(self, q):
        for i in range(len(q)):
            self.jointlinks[i].q = q[i]

    def parent(self, i):
        return self.parent_idx[i]

    def update(self):
        NB = self.NB
        dim = self.dim
        for i in range(NB):
            I  = self.jointlinks[i].I
            XJ = self.jointlinks[i].XJ()
            j = self.parent(i)
            if j != -1: # parent is not root
                XT = self.jointlinks[j].XT
            else:
                XT = np.eye(dim)
            X_j_to_i = XJ @ XT
            self.X_parent_to[i] = X_j_to_i

            if j != -1: # parent is not root
                self.jointlinks[i].X_r_to = X_j_to_i @ self.jointlinks[j].X_r_to
                velj = self.vel[j]
            else:
                self.jointlinks[i].X_r_to = X_j_to_i
                velj = self.velb

            vJ = self.jointlinks[i].vJ()
            self.vel[i] = X_j_to_i @ velj + vJ

    # recursive newton-euler on each body coordinate
    def inverse_dynamics(self, ddq, fext):
        q = self.q()
        dq = self.dq()
        NB = self.NB
        dim = self.dim

        assert NB == len(ddq)
        assert NB == len(fext)
        assert (NB,dim) == fext.shape

        acc = np.zeros((NB, dim))
        tau = np.zeros(NB)
        f = np.zeros((NB, dim))

        S = [self.jointlinks[i].S() for i in range(NB)]
        np.zeros((NB, dim, dim))

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

            acc[i] = X_j_to_i @ accj  + S[i] * ddq[i] + cJ + crm(self.vel[i]) @ vJ

            X_r_to = self.jointlinks[i].X_r_to

            if self.use_psudo_acc:
                f[i] = I @ acc[i] + crf(self.vel[i]) @ I @ self.vel[i] - dualm(X_r_to) @ fext[i]
            else:
                accg_i = X_r_to @ np.array([0, 0, -self.g_acc])
                f[i] = I @ (acc[i] - accg_i)+ crf(self.vel[i]) @ I @ self.vel[i] - dualm(X_r_to) @ fext[i]

        for i in range(NB-1, -1, -1):
            tau[i] = S[i] @ f[i]
            j = self.parent(i)
            if j != -1: # parent is root
                X_j_to_i = self.X_parent_to[i]
                f[j] = f[j] + X_j_to_i.T @ f[i]

        return tau


    def composite_inertia(self, dtype=np.float64):

        X_parent_to = self.X_parent_to
        NB = self.NB
        dim = self.dim
        H = np.zeros([NB, NB], dtype=dtype)
        S = [self.jointlinks[i].S() for i in range(NB)]

        self.Ic = [np.zeros([dim, dim]) for i in range(NB)]
        for i in range(NB):
            self.Ic[i] = self.jointlinks[i].I

        for i in range(NB-1, -1, -1):
            j = self.parent(i)
            if j != -1: # parent is root
                X_j_to_i = X_parent_to[i]
                #self.Ic[j] = self.Ic[j] + X_j_to_i.T @ self.jointlinks[i].I @ X_j_to_i
                self.Ic[j] = self.Ic[j] + X_j_to_i.T @ self.Ic[i] @ X_j_to_i
            #print(i, self.Ic[i])
            F = self.Ic[i] @ S[i]
            H[i][i] = S[i].T @ F

            j = i
            while self.parent(j) != -1:
                F = X_parent_to[j].T @ F
                j = self.parent(j)
                H[i][j] = F.T @ S[j]
                H[j][i] = H[i][j]

        # local to global
        #for i in range(NB):
        #    self.Ic[i] = self.jointlinks[i].X_r_to.T @ self.Ic[i] @ self.jointlinks[i].X_r_to

        return H


    def foward_dynamics(self, fext=None):
        if fext is None:
            fext = np.zeros((self.NB, self.dim))
        tau = np.array([jl.active_force() for jl in self.jointlinks])
        C = self.inverse_dynamics(np.zeros(self.NB), fext)
        H = self.composite_inertia()
        ddq = np.linalg.solve(H, tau - C)
        return ddq

    def step_with_ddq(self, ddq, dt):
        # simple leap frog for spring
        # ode45 is better?
        for i in range(self.NB):
            self.jointlinks[i].dq = self.jointlinks[i].dq + ddq[i] * dt
            self.jointlinks[i].q = self.jointlinks[i].q + self.jointlinks[i].dq * dt
        self.update()

    def step(self, dt, fext=None):
        ddq = self.foward_dynamics(fext)
        self.step_with_ddq(ddq, dt)

    def drawCmds(self):
        cmds = []
        for i in range(self.NB):
            cmds = cmds + self.jointlinks[i].drawCmd()
        return cmds

    #def momentum(self):
    #    # (2.60) ho = dual(Xc) hc
    #    # => h = dual(X) ho
    #    h = np.zeros(self.dim)
    #    for i in range(self.NB):
    #        X_r_to_i = self.X_r_to[i]
    #        v = self.vel[i]
    #        I = self.jointlinks[i].I
    #        # dual_invert = transpose
    #        X_i_to_r_dual = X_r_to_i.T
    #        h = h + X_i_to_r_dual @ I @ self.vel[i]
    #    return h

    # q H q is also ok
    def kinetic_energy(self):
        T = 0
        for i in range(self.NB):
            v = self.vel[i]
            I = self.jointlinks[i].I
            #Xc = self.jointlinks[i].Xc
            #Ic = self.jointlinks[i].Ic
            #vc = Xc @ v
            T = T + (v @ I @ v) / 2
            #T = T + (vc @ Ic @ vc)/2
        return T

    def potential_energy(self):
        U = 0
        for i in range(self.NB):
            X_r_to_i = self.jointlinks[i].X_r_to
            _, _, _, y = Xtoscxy(self.jointlinks[i].Xc @ X_r_to_i)
            U = U + y * self.g_acc * self.jointlinks[i].m
        return U

    def syn_inertia_info(self):
        I = self.Ic[0][0,0]
        m = self.Ic[0][1,1]
        cog = np.array([self.Ic[0][0,2]/m, -self.Ic[0][0,1]/m])
        return I, m, cog

def ref(q, dq, I1, I2):
    m1 = 1
    m2 = 1
    l1 = 2
    l2 = 1
    q1 = q[0]
    q2 = q[1]
    dq1 = dq[0]
    dq2 = dq[1]
    g = 9.8
    H = np.zeros((2,2))
    H[0][0] = (4*l1*l2*m2*cos(q2)+(l2**2+4*l1**2)*m2+l1**2*m1+4*I2+4*I1)/4.0E+0
    H[0][1] = (2*l1*l2*m2*cos(q2)+l2**2*m2+4*I2)/4.0E+0
    H[1][0] = (2*l1*l2*m2*cos(q2)+l2**2*m2+4*I2)/4.0E+0
    H[1][1] = (l2**2*m2+4*I2)/4.0E+0

    C = np.zeros(2)
    C[0] = (-(dq2**2*l1*l2*m2*cos(q1)*sin(q2+q1))/2.0E+0)-dq1*dq2*l1*l2*m2*cos(q1)*sin(q2+q1)+(dq2**2*l1*l2*m2*sin(q1)*cos(q2+q1))/2.0E+0+dq1*dq2*l1*l2*m2*sin(q1)*cos(q2+q1)+(g*l2*m2*cos(q2+q1))/2.0E+0+g*l1*m2*cos(q1)+(g*l1*m1*cos(q1))/2.0E+0
    C[1] = (dq1**2*l1*l2*m2*cos(q1)*sin(q2+q1))/2.0E+0-(dq1**2*l1*l2*m2*sin(q1)*cos(q2+q1))/2.0E+0+(g*l2*m2*cos(q2+q1))/2.0E+0

    return np.linalg.solve(H, -C)


def test_link():
    K = 100
    #jl1 = StickJointLink(1, 2, RevoluteJoint(), q=30*np.pi/180)
    jl1 = StickJointLink(1, 2, RevoluteJoint(), q=30*np.pi/180)
    jl2 = StickJointLink(1, 1, RevoluteJoint(), q=30*np.pi/180)
    #jl2 = StickSpringJointLink(1, 1, K, RevoluteJoint(), q=30*np.pi/180)
    I1 = jl1.Ic[0][0]
    I2 = jl2.Ic[0][0]
    model = LinkTreeModel([jl1, jl2])

    import graphic
    viewer = graphic.Viewer(scale=100, offset=[0, -0.1])
    #dt = 0.005
    t = 0
    dt = 0.001
    while True:
        t = t + dt
        q = model.q()
        dq = model.dq()
        #if True:
        if False:
            ddq = ref(q, dq, I1, I2)
        else:
            ddq = model.foward_dynamics()
        model.step_with_ddq(ddq, dt)
        k = model.kinetic_energy()
        u = model.potential_energy()
        #print(model.momentum())
        viewer.clear()
        viewer.handle_event(graphic.default_event_handler)
        viewer.text([f"{t:.03f} : {q}"])
        viewer.draw(model.drawCmds())
        viewer.flush()

if __name__ == '__main__':
    test_link()

