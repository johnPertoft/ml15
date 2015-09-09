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
	return polynomialKernel(x,y, 2)

def buildMatrix(data):
	P = [0] * len(data)
	for i in range(0, len(data)):
		P[i] = [0] * len(data)
		for j in range(0, len(data)):
			P[i][j] = data[i][2] * data[j][2] * kernel(numpy.array(data[i][0:2]), numpy.array(data[j][0:2]))

	return P

def indicator(data, alpha, x):
	s = 0
	for i in range(0, len(data)):
		s = s + alpha[i] * data[i][2]*kernel(numpy.array(x), numpy.array(data[i][0:2]))
	return s
# ----------- Start

numA = 10
numB = numA * 2
classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(numA)] + [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(numA)]

classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(numB)]

data = classA + classB

random.shuffle(data)

pylab.hold(True)
pylab.plot([p[0] for p in classA], [p[1] for p in classA],
	'bo')
pylab.plot([p[0] for p in classB], [p[1] for p in classB],
	'ro')

P = matrix(buildMatrix(data))
q = matrix(0.0, (len(data), 1))
h = matrix(-1.0, (len(data), 1))
G = numpy.identity(len(data)) * -1

r = qp(matrix(P), q, matrix(G), h)
alpha = list(r['x'])
alpha = filter(lambda a: a > 0.00001, alpha) # remove zeros

x_range = numpy.arange(-4, 4, 0.05)
y_range = numpy.arange(-4, 4, 0.05)

grid = matrix([[indicator(data, alpha, [x, y]) for y in y_range] for x in x_range])

pylab.contour(x_range, y_range, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1,3,1))
pylab.show()


