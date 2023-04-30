from arith import *
import numpy as np

#-------------------------
# Geometory
#-------------------------

class GeoLineSegment():
    def __init__(self, l):
        self.l = l

    def drawCmd(self, X):
        s, c, x, y = Xtoscxy(X)
        dir = np.array([c,s])
        org = np.array([x,y])
        return [{"type": "lineseg", "start":org, "end":org+dir*self.l}]


class GeoCircle():
    def __init__(self, r):
        self.r = r

    def drawCmd(self, X):
        s, c, x, y = Xtoscxy(X)
        return [ {"type": "circle", "origin":(x,y), "r":self.r}]

def test_draw():
    import graphic
    viewer = graphic.Viewer()

    X1 = Xpln(np.pi/4, np.array([100,2]))
    cmd1 = GeoLineSegment(10).drawCmd(X1)

    X2 = Xpln(-np.pi/4, np.array([-200,20]))
    cmd2 = GeoCircle(10).drawCmd(X2)

    while True:
        viewer.clear()
        viewer.handle_event(graphic.default_event_handler)
        viewer.draw(cmd1)
        viewer.draw(cmd2)
        viewer.flush(0.03)


if __name__ == '__main__':
    test_draw()

