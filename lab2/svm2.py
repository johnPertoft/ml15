import cvxopt
from cvxopt.solvers import qp
from cvxopt.base import matrix

import numpy as np, pylab, random, math, sys
from numpy import linalg as LA

def linearKernel(X, Y):
    return X.transpose() * Y + 1

def kernel(X):
    #return linearKernel(x,y)

# TODO: do matrix mult instead
def ind(X, t, alpha, x):
    s = 0
    for i in range(len(alpha)):
        s += alpha[i] * t[i] * kernel(X[i, :], x)

    return s

# ----------- Start

useSlack = False
C = 0
if len(sys.argv) >= 2:
    C = float(sys.argv[1])
    useSlack = True

classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)] + [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)]

classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(10)]

data = classA + classB
random.shuffle(data)

X = np.array([[d[0], d[1]] for d in data]) 
t = np.array([d[2] for d in data])

q = -1 * np.ones(len(data))
h = np.zeros(len(data))
G = -1 * np.eye(len(data))
if useSlack:
    # append to form the equation with the new constraints
    h = np.hstack((h, C * np.ones(len(data))))
    G = np.vstack((G, np.eye(len(data)))) 

N = len(X)
# TODO: do kernels as matrix operation instead
P = np.outer(t, t) * kernel(X)

r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
alpha = np.array(r['x'])[:, 0]
svs = np.nonzero(np.absolute(alpha) > 1e-6)[0]

x_range = np.arange(-4, 4, 0.05)
y_range = np.arange(-4, 4, 0.05)

grid = matrix(
        [[ind(X[svs], t[svs], alpha[svs], np.array([x, y]))
            for y in y_range] for x in x_range])

pylab.hold(True)
pylab.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
pylab.plot([p[0] for p in classB], [p[1] for p in classB], 'ro')

pylab.contour(x_range, y_range, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))
pylab.show()
