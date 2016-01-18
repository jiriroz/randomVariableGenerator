"""
Random variable generator for discrete and continuous distributions.

Discrete distributions:
poisson
        
Continuous distributions:
normal
m-variate normal
exponential

"""

import random
import math
import sys
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
from mpl_toolkits.mplot3d import axes3d

"""
Generate random variables.
"""
class RVGenerator:

    def __init__(self):
        self.normalCached = None

    """
    Returns a random poisson variable with parameter lambda.
    """
    def poisson(self, lambd=1.0):
        if lambd <= 0:
            raise ValueError("Lambda needs to be > 0")
        return self._poissonKnuth(lambd=float(lambd))

    """
    Generates a random poisson variable based on the pdf for poisson
    distribution.
    """
    def _poisson(self, lambd=1.0):
        u = random.random()
        p = math.e**(-lambd)
        s = p
        i = 0
        while s < u:
            i += 1
            p *= (lambd/i)
            s += p
        return i

    """
    Generates a random poisson variable based on Knuth's algorithm.
    """
    def _poissonKnuth(self, lambd=1.0):
        s, k, p = math.e**(-lambd), 0, 1
        while True:
            k += 1
            u = random.random()
            p *= u
            if s > p:
                break
        return k - 1

    """
    Returns a random normal variable with parameters (mu, sigma2)
    generated with Box-Muller transform.
    """
    def normal(self, mu=0.0, sigma2=1.0):
        mu, sigma2 = float(mu), float(sigma2)
        if self.normalCached == None:
            sigma = math.sqrt(sigma2)

            x1 = random.random()
            x2 = random.random()
            y1 = math.sqrt(-2 * math.log(x1)) * math.cos(2 * math.pi * x2)
            y2 = math.sqrt(-2 * math.log(x1)) * math.sin(2 * math.pi * x2)
            self.normalCached = y2*sigma + mu
            return y1*sigma + mu
        else:
            temp = self.normalCached
            self.normalCached = None
            return temp

    """
    Return a pair of normal random variables with given means and covariances.
    """
    def bivar_normal(self, mu1=0, mu2=0, var1=1, var2=1, cov=0):
        return self.multivar_normal(mu=[mu1, mu2], cov=[[var1, cov],[cov, var2]])

    """
    Given a vector mu and a covariance matrix, returns a random
    sample from the normal distribution.
    """
    def multivar_normal(self, mu=[0.0], cov=[[1.0]]):
        n = len(mu)
        mu = np.array(mu, dtype=np.float32)
        cov = np.array(cov, dtype=np.float32)
        A = self._cholesky(cov)
        z = [self.normal() for x in range(n)]
        return np.add(mu, np.dot(A, z))

    """
    Given a matrix A, find a matrix T such that T * transpose(T) = A.
    This process is called Cholesky decomposition.
    The matrix A is a covariance, positive semi-definite matrix.
    The matrix T is used in drawing random values from a multivariate
    random normal distribution.
    """
    def _cholesky(self, A):
        L = [[0.0] * len(A) for x in range(len(A))]
        for i in range(len(A)):
            for j in range(i+1):
                s = sum([L[i][k] * L[j][k] for k in range(j)])
                L[i][j] = math.sqrt(A[i][i] - s) if (i == j) else \
                    (1.0 / L[j][j] * (A[i][j] - s))
        return np.array(L)

    """
    Generate an exponential random variable.
    """
    def exp(self, lambd=1.0):
        if lambd <= 0:
            raise ValueError("Lambda needs to be > 0")
        return self._exp(lambd=float(lambd))

    """
    Returns an exponential random number.
    """
    def _exp(self, lambd=1.0):
        return -math.log(1 - random.random()) / lambd

"""
Draw n values from a specified univariate distribution
and plot a histogram of the drawn values.
"""
def monteCarlo(func, n=1000, line=False, **args):
    vals = [func(**args) for i in range(n)]
    fig, ax = plt.subplots()
    #the histogram of the data
    n, bins, patches = ax.hist(vals, int(20*math.log(n,10)), normed=1, facecolor='green', alpha=0.7)
    d = bins[1] - bins[0]
    bins = np.array(bins[1:]) - d/2 #center the line
    if line:
        ax.plot(bins, n, color='black',linewidth=2.0)
    lower = min(vals)
    upper = max(vals)
    ax.axis([lower, upper, 0, 1])
    ax.grid(True)
    plt.show()

"""
Draw n values from a specified bivariate distribution
and plot a 3D histogram of the result.
"""
def monteCarloBivar(func, n=1000, **args):
    vals = np.array([func(**args) for i in range(n)]).T
    x = vals[0]
    y = vals[1]
    bins = 20
    H, xedges, yedges = np.histogram2d(x,y,bins=bins,normed=True)
    H = np.array(H)
    xmin, xmax = xedges[0], xedges[-1]
    ymin, ymax = yedges[0], yedges[-1]
    X = np.linspace(xmin, xmax, num=bins)
    Y = np.linspace(ymin, ymax, num=bins)
    X = np.array([X] * bins)
    Y = np.array([Y] * bins).T
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, H, rstride=1, cstride=1, color="red")
    plt.show()

"""
Draw n values from a specified trivariate distribution and
plot a scatterplot of the result. n > 1000 not recommended, since
plots every data point.
"""
def monteCarloTrivar(func, n=100, **args):
    vals = np.array([func(**args) for i in range(n)]).T
    x = vals[0]
    y = vals[1]
    z = vals[2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, linewidths=1)
    plt.show()

if __name__ == "__main__":
    gen = RVGenerator()
    monteCarlo(gen.normal, n=10000, mu=5.0, sigma2=0.7)
    monteCarloBivar(gen.bivar_normal, mu1=0, mu2=5, var1=0.5, var2=10, cov=0.9, n=10000)
    monteCarloTrivar(gen.multivar_normal, n=1000, mu=[0,0,0], cov=[[1,0.8,0.5],[0.8,1,0],[0.5,0,1]])
