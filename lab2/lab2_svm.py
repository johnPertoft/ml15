import cvxopt
from cvxopt.solvers import qp
from cvxopt.base import matrix

import numpy, pylab, random, math

def linearKernel(x, y):
   return x.transpose() * y + 1;

def polynomialKernel(x, y, p):
    return numpy.power(x.transpose() * y + 1, p)

def rbfKernel(x, y, sigma):
    return numpy.exp(-1 * numpy.power(x - y, 2) / (2 * sigma * sigma))

def sigmoidKernel(x, y, k, delta):
    return numpy.tanh(k * x.transpose() * y - delta)

print "hello"
