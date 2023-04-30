from sympy import *

#init_printing(use_unicode=True)

#-------------------------
# Arith
#-------------------------

# scalar product: *
# matrix product: *
# matrix-vector product: *

# General Transform : 2.24~2.27
def inversem(X):
    c = X[1,1]
    s = X[1,2]
    x = s*X[1,0] + c*X[2,0]
    y = -c*X[1,0] + s*X[2,0]
    return Matrix([
        [1, 0, 0],
        [y, c, -s],
        [-x, s, c]
        ])

def inversef(X):
    c = X[1,1]
    s = -X[1,2]
    x = -X[2,0]
    y = X[1,0]
    return Matrix([
        [1, 0, 0],
        [s * r[0] - c * r[1], c, s],
        [c * r[0] + s * r[1], -s, c]
        ])

def dualm(X):
    return inversem(X).T

def dualf(X):
    return inversef(X).T

# planar coordinate transform matrix : M -> M
# r : x, y
# for pos, not vel,acc
# X * r means convert r in dst(X)-coordinate into X*r in original-coordinate
def Xpln(th, r):
    c = cos(th)
    s = sin(th)
    return Matrix([
        [1, 0, 0],
        [s * r[0] - c * r[1], c, s],
        [c * r[0] + s * r[1], -s, c]
        ])

# NOTE:
# vector does not express "position"
# position can be expressed by velocity (unit time)??
# but origin-offset is not expressed.
# position can be represented by X
#
#def vec(x, y):
#    return Matrix([
#        [0],
#        [x],
#        [y],
#        ])
    # XtoV for th = 0
    #return Matrix([
    #    [X[1][2]],
    #    [X[2][0] + X[1][1]*X[2][0] + X[1][2]*X[1][0]],
    #    [-X[1][0] - X[1][1]*X[1][0] + X[1][2]*X[2][0]],
    #    ])

def Xtoscxy(X):
    #th = np.arctan2(X[1][2], X[1][1])
    x = X[1][0] * X[1][2] + X[2][0] * X[2][2]
    y = -X[1][0] * X[2][2] + X[2][0] * X[1][2] 
    return ([X[1][2], X[1][1], x, y])

# planar cross for motion
def crm(v):
    omega = v[0]
    vx = v[1]
    vy = v[2]
    return Matrix([
        [0, 0, 0],
        [vy, 0, -omega],
        [-vx, omega, 0]
        ])


# planar cross for force
def crf(v):
    omega = v[0]
    vx = v[1]
    vy = v[2]
    return Matrix([
        [0, -vy, vx],
        [0, 0, -omega],
        [0, omega, 0]
        ])



if __name__ == '__main__':
    def skew_space(v):
        return Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    q1, r = symbols('q1 r')
    E = Matrix([[cos(q1), sin(q1), 0], [-sin(q1), cos(q1), 0], [0, 0, 1]])
    p = Matrix([[r*q1],[0],[0]])
    tmp1 = zeros(6)
    tmp1[:3,:3] = E
    tmp1[3:,3:] = E
    tmp2 = eye(6)
    tmp2[3:,:3] = -skew_space(p)
    #pprint(tmp2)
    #print(tmp2[2:5,2:5])
    pprint((tmp1 * tmp2)[2:5,2:5])
    print((tmp1 * tmp2)[2:5,2:5])
    #dL = diff(Ls, 1, dt)

