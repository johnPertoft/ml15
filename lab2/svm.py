import cvxopt
from cvxopt.solvers import qp
from cvxopt.base import matrix

import numpy, pylab, random, math

def linearKernel(x, y):
    return x.dot(y) + 1;

def polynomialKernel(x, y, p):
    return numpy.power(x.dot(y) + 1, p)

def rbfKernel(x, y, sigma):
    return numpy.exp(-1 * numpy.power(x - y, 2) / (2 * sigma * sigma))

def sigmoidKernel(x, y, k, delta):
    return numpy.tanh(k * x.transpose() * y - delta)

def kernel(x, y):
    #return linearKernel(x,y)
    return polynomialKernel(x, y, 2)

def kernels(X):
    k = numpy.array(numpy.zeros([len(X), len(X)]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            k[i, j] = kernel(X[i, :], X[j, :])

    return k

def indicator(data, alpha, x):
    s = 0
    for i in range(0, len(data)):
        s = s + alpha[i] * data[i][2]*kernel(numpy.array(x), numpy.array(data[i][0:2]))

    return s

# ----------- Start

classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)] + [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)]

classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(10)]

data = classA + classB
random.shuffle(data)

X = numpy.array([[d[0], d[1]] for d in data]) 
t = numpy.array([d[2] for d in data])

q = -1 * numpy.ones(len(data))
h = numpy.zeros(len(data))
G = -1 * numpy.eye(len(data))

P = numpy.outer(t, t) * kernels(X)
r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
alpha = list(r['x'])
alpha = filter(lambda a: a > 1e-6, alpha)
x_range = numpy.arange(-4, 4, 0.05)
y_range = numpy.arange(-4, 4, 0.05)

grid = matrix([[indicator(data, alpha, [x, y]) for y in y_range] for x in x_range])

pylab.hold(True)
pylab.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
pylab.plot([p[0] for p in classB], [p[1] for p in classB], 'ro')

pylab.contour(x_range, y_range, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))
pylab.show()


