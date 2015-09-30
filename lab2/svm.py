import cvxopt
from cvxopt.solvers import qp
from cvxopt.base import matrix

import numpy as np, pylab, random, math, sys
from numpy import linalg as LA

# +1 to the dot product to account for the bias term what was added to W, ??
def linearKernel(x, y):
    return x.dot(y) + 1

def polynomialKernel(x, y, p):
    return np.power(x.dot(y) + 1, p)

def rbfKernel(x, y, sigma):
    return np.exp(-1 * np.power(LA.norm(x - y) , 2) / (2 * sigma * sigma))

def sigmoidKernel(x, y, k, delta):
    return np.tanh(k * (x.dot(y) + 1) - delta)

def kernel(x, y):
    #return linearKernel(x,y)
    #return polynomialKernel(x, y, 3)
    return rbfKernel(x, y, 1.0)
    #return sigmoidKernel(x, y, 0.5, 0)

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

"""
with these parameters, cvxopt minimizes
0.5 * (x^T)Px + (q^T)x
subject to Gx <= h

we have the optimization problem
0.5 * (a^T)Pa - (a^T)(1, 1, 1, ...)^T
subject to
a_i >= 0 for all i

thus
x = a, the alphas that we want to find
h = zeros(len(data))
G = -1 * eye(len(data)) (minus because the contraint is flipped)

(P_ij = t_i * t_j * K(x_i, x_j))

When adding slack variables we add some contraints to the dual form so that
we have
a_i >= 0 for all i (from before)
a_i <= C

Thus
h = [zeros(len(data)), C * ones(len(data))]^T
G = [-1 * eye(len(data))
     eye(len(data))]

"""

q = -1 * np.ones(len(data))
h = np.zeros(len(data)) # alpha_i >= 0 constraints
G = -1 * np.eye(len(data))
if useSlack:
    h = np.hstack((h, C * np.ones(len(data)))) # alpha_i <= C constraints
    G = np.vstack((G, np.eye(len(data)))) 

N = len(X)
P = np.outer(t, t) * np.array(
        [[kernel(X[i, :], X[j, :]) for j in range(N)] for i in range(N)])

r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
alpha = np.array(r['x'])[:, 0]
svs = np.nonzero(np.absolute(alpha) > 1e-6)[0]
print "Support vectors: ", svs

x_range = np.arange(-4, 4, 0.05)
y_range = np.arange(-4, 4, 0.05)

grid = matrix(
        [[ind(X[svs], t[svs], alpha[svs], np.array([x, y]))
            for y in y_range] for x in x_range])

pylab.hold(True)
pylab.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
pylab.plot([p[0] for p in classB], [p[1] for p in classB], 'ro')

pylab.contour(x_range, y_range, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))

## test svm on testdata
testA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(50)] + [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(50)]

testB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(100)]

testData = testA + testB
random.shuffle(testData)
Ttest = np.array([d[2] for d in testData])
c = np.array([ind(X[svs], t[svs], alpha[svs], np.array([d[0], d[1]])) for d in testData])
c = np.sign(c)
c = ((c * Ttest) > 0)
corr = (100.0 * np.sum(c)) / len(testData)
print "Correctly classified test samples: ", corr, "%"
pylab.plot([p[0] for p in testA], [p[1] for p in testA], 'b+')
pylab.plot([p[0] for p in testB], [p[1] for p in testB], 'r+')
pylab.show()

