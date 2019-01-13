from matplotlib import pyplot as plt
import seaborn
from .ssm import *
from particles.core import SMC


def simulate_plot(model):
    x, y = model.simulate(41)
    y = np.asarray(y)[:, 0, :]
    plt.plot(np.exp(x))
    plt.plot(y)


def plot_theta(prior, model, burnin=False, m=None, linecolor='darkred'):
    """
    Plot the Markov chain obtained in the HM for all parameters defined in the prior dictionnary
    """
    for p in prior.keys():
        plt.figure()
        plt.plot(model.chain.theta[p])
        if burnin:
            plt.axvline(x=m[0], color=linecolor, linestyle='--')
            plt.axvline(x=m[1], color=linecolor, linestyle='--')
        plt.xlabel('iter')
        plt.ylabel(p)
    plt.figure()
    plt.plot(model.chain.lpost)
    if burnin:
        plt.axvline(x=m[0], color=linecolor, linestyle='--')
        plt.axvline(x=m[1], color=linecolor, linestyle='--')
    plt.xlabel('iter')
    plt.ylabel('lpost')
    plt.show()


def print_metrics(model):
    """
    Print a few interesting metrics about the HM
    """
    print('mean square jump distance: {}'.format(model.mean_sq_jump_dist(discard_frac=0.1)))
    print('posterior loglikelihood: {}'.format(model.chain.lpost[-5:]))
    print('Acceptance rate: {}'.format(model.acc_rate))
    print('Last terms of theta chain: {}'.format(model.chain.theta[-3:]))


def distplot(prior, model, start):
    """
    Plot the marginal distribution of all of the parameters in the HM after discarding the steps before the start
    """
    for p in prior.keys():
        plt.figure()
        seaborn.distplot(model.chain.theta[p][start:]).set_title(p)
    plt.show()


def get_trajectories(N, start, model, pmmh, n_particles=10000):
    """
    Posterior sampling to get N samples from the chosen model using a (new) particle filter
    """
    simul = np.zeros((N, 41))
    for t in range(N):
        param = np.random.choice(pmmh.chain.theta[start:])
        if model == 'poisson':
            my_model = RandomWalk2D_poisson(sigma=param['sigma'])
        elif model == 'RW':
            my_model = RandomWalk2D(sigma=param['sigma'], tau=param['tau'])
        elif model == 'LDrift':
            my_model = LDPDrift(sigma=param['sigma'], tau=param['tau'], r=param['r'])
        else:
            my_model = LogisticDiffusion2D(sigma=param['sigma'], tau=param['tau'], r=param['r'], b=param['b'])
        fk_model = ssm.Bootstrap(ssm=my_model, data=y)
        pf = SMC(fk=fk_model, N=n_particles, moments=True, store_history=True)
        pf.run()
        simul[t] = np.exp(pf.hist.backward_sampling(1, linear_cost=False))
    return simul


def plot_posterior_trajectories(traj):
    """
    Plot the trajectories along with the values of the countings
    """
    X = data['time']
    plt.plot(X, traj.mean(axis=0))
    q = np.quantile(traj, [0.025, 0.975], axis=0)
    plt.fill_between(X, q[0], q[1], color='lightblue')
    plt.scatter(X, data['C0'])
    plt.scatter(X, data['C1'])
    plt.show()