import copy
import particles.distributions as dists
import numpy as np
from scipy import stats
import numpy.random as random


class LDP(dists.ProbDist):  # A first distribution that is needed to sample X
    """Logistic Diffusion process"""
    def __init__(self, loc=0., r=0., b=0., sigma=1., dt=1., step=100.):
        self.r = r
        self.b = b
        self.sigma = sigma
        self.step = step
        self.dt = dt
        self.loc = loc

    def rvs(self, size=1):
        x = copy.copy(self.loc)
        count = self.dt*self.step
        count, remain = int(count), (count - int(count))/self.step
        W = np.random.normal(size=count)
        for i in range(count):
            x += (self.r + self.sigma ** 2 / 2 - self.b * np.exp(x)) / self.step + self.sigma / np.sqrt(self.step) * W[
                i]
        if remain != 0:
            x += (self.r + self.sigma ** 2 / 2 - self.b * np.exp(x)) * remain + self.sigma * np.sqrt(
                remain) * np.random.normal()
        return x


class NegativeBinomial(dists.DiscreteDist):
    def __init__(self,n,p):
        self.n=n
        self.p=p

    def rvs(self, size=1):
        return random.negative_binomial(self.n,self.p, size=size)

    def logpdf(self, x):
        return stats.nbinom.logpmf(x, self.n,self.p)

    def ppf(self, u):
        return stats.nbinom.ppf(u,self.n, self.p)
