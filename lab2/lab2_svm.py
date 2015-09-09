import cvxopt
from cvxopt.solvers import qp
from cvxopt.base import matrix

import numpy, pylab, random, math

def linearKernel(x, y):
   return x.dot(y) + 1;

def polynomialKernel(x, y, p):
    return numpy.power(x.transpose() * y + 1, p)

def rbfKernel(x, y, sigma):
    return numpy.exp(-1 * numpy.power(x - y, 2) / (2 * sigma * sigma))

def sigmoidKernel(x, y, k, delta):
    return numpy.tanh(k * x.transpose() * y - delta)

def kernel(x, y):
	return linearKernel(x,y)

def buildMatrix(data):
	P = [0] * len(data)
	for i in range(0, len(data)):
		P[i] = [0] * len(data)
		for j in range(0, len(data)):
			P[i][j] = data[i][2] * data[j][2] * kernel(numpy.array(data[i][0:2]), numpy.array(data[i][0:2]))

	return P

# ----------- Start

classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)] + [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)]

classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(10)]

data = classA + classB

random.shuffle(data)

pylab.hold(True)
pylab.plot([p[0] for p in classA], [p[1] for p in classA],
	'bo')
pylab.plot([p[0] for p in classB], [p[1] for p in classB],
	'ro')

P = buildMatrix(data)
q = matrix(numpy.array(len(data) * [-1]))

print q
h = [0] * len(data)
G = numpy.diag([-1] * len(data))

print matrix(P)

r = qp(matrix(P), q, matrix(G), matrix(h))
alpha = list(r['x'])
print alpha
pylab.show()


