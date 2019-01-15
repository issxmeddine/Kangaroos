import particles.state_space_models as ssm
import particles.distributions as dists
import numpy as np
import pandas as pd
from .dists import LDP, NegativeBinomial


############################### Global variables  ####################################
data = pd.read_csv("../../Data/data.csv",index_col=0)
data.columns = ['C0','C1','time']

dt = data['time'].diff()
y = np.array(data[['C0', 'C1']])


def y_to_list(y):
    y_list = []
    for i in range(y.shape[0]):
        y_list.append(np.array([y[i]]))
    return (y_list)


y = y_to_list(y)

mean_N0 = y[0].mean()
std_N0 = y[0].std()

############################### Simplest model possible ###############################
class RandomWalk2D_poisson(ssm.StateSpaceModel):
    '''
    A Random Walk Model with Poisson emission distribution
    '''
    default_parameters = {'sigma': 0.01}

    def PX0(self):
        return dists.Normal(loc=np.log(mean_N0) ,scale=0.01)

    def PX(self, t, xp):
        return dists.Normal(loc=xp ,scale=self.sigma *np.sqrt(dt[t]))

    def PY(self, t, xp, x):
        return dists.IndepProd(dists.Poisson(rate=np.exp(x)),
                               dists.Poisson(rate=np.exp(x)))


####################################### Model M3 ######################################
class RandomWalk2D(ssm.StateSpaceModel):
    '''
    Random Walk Model (M3)
    '''
    default_parameters = {'sigma': 0.01, 'tau': 0.001}

    def PX0(self):
        return dists.Normal(loc=np.log(mean_N0), scale=0.01)

    def PX(self, t, xp):
        return dists.Normal(loc=xp, scale=self.sigma * np.sqrt(dt[t]))

    def PY(self, t, xp, x):
        return dists.IndepProd(NegativeBinomial(n=1 / self.tau, p=1 / (1 + self.tau * np.exp(x))),
                               NegativeBinomial(n=1 / self.tau, p=1 / (1 + self.tau * np.exp(x))))


####################################### Model M2 ######################################
class LDPDrift(ssm.StateSpaceModel):
    '''
    Logistic Diffusion process with drift (M2)
    '''
    default_parameters = {'sigma': 0.01, 'tau': 0.001, 'r': 0.005}

    def PX0(self):
        return dists.Normal(loc=np.log(mean_N0), scale=0.01)

    def PX(self, t, xp):
        return dists.Normal(loc=xp + self.r * dt[t], scale=self.sigma * np.sqrt(dt[t]))

    def PY(self, t, xp, x):
        return dists.IndepProd(NegativeBinomial(n=1 / self.tau, p=1 / (1 + self.tau * np.exp(x))),
                               NegativeBinomial(n=1 / self.tau, p=1 / (1 + self.tau * np.exp(x))))


####################################### Model M1 ######################################

# No discretization
class LogisticDiffusion2D(ssm.StateSpaceModel):
    '''
    Logistic Diffusion process with drift and restoring force: No Euler discretization (M1)
    '''
    default_parameters = {'r': 0., 'b': 0., 'sigma': 0.01, 'tau': 0.001}

    def PX0(self):
        return dists.Normal(loc=np.log(mean_N0), scale=0.01)

    def PX(self, t, xp):
        loc = xp + (self.r - self.sigma ** 2 / 2 + self.b * np.exp(xp)) * dt[t]
        return dists.Normal(loc=loc, scale=self.sigma * np.sqrt(dt[t]))

    def PY(self, t, xp, x):
        return dists.IndepProd(NegativeBinomial(n=1 / self.tau, p=1 / (1 + self.tau * np.exp(x))),
                               NegativeBinomial(n=1 / self.tau, p=1 / (1 + self.tau * np.exp(x))))


# Discretization
class LDEuler(ssm.StateSpaceModel):
    '''
    Logistic Diffusion process with Euler Discretization (M1)
    '''
    default_parameters = {'r': 0., 'b': 0., 'sigma': 0.01, 'tau': 0.001}

    def PX0(self):
        return dists.Normal(loc=np.log(mean_N0), scale=0.01)

    def PX(self, t, xp):
        if self.r<= 0:
            return LDP(loc=xp, r=self.r, b=self.b, sigma=self.sigma, dt=dt[t], step=100.)
        else:
            return dists.Gamma(a=2*self.r/self.sigma**2, b=2*self.b/self.sigma**2)  # gamma stationary distribution

    def PY(self, t, xp, x):
        return dists.IndepProd(NegativeBinomial(n=1 / self.tau, p=1 / (1 + self.tau * np.exp(x))),
                               NegativeBinomial(n=1 / self.tau, p=1 / (1 + self.tau * np.exp(x))))
