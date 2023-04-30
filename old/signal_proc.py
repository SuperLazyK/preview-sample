import numpy as np

def diff(xs, n, dt):
    lastn = xs[-(n+1):]
    if n == 1:
        return (lastn[1] - lastn[0]) / dt
    elif n == 2:
        return (lastn[2] + lastn[0] - 2*lastn[1]) / (dt**2)
    else:
        assert False, "not implemented"

#reversed-time lpf
def rlpf(xs, Tc, dt):
    y = xs[-1]
    a = dt / (Tc + dt)
    for i in range(2,len(xs)+1):
        y = a * xs[-i] + (1 - a) * y
    return y

def lpf(xs, T, dt):
    y = xs[0]
    a = dt / (T + dt)
    ret = [y]
    for i in range(1,len(xs)):
        y = a * xs[i] + (1 - a) * y
        ret.append(y)
    return ret

def sample(pfs, t):
    for f,r in pfs:
        if r[0] <= t and t < r[1]:
            return f(t)
    assert False, t

