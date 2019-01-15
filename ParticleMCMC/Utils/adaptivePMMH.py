from scipy import stats
from sklearn import cluster, mixture
import numpy as np
import particles.mcmc as mcmc


class AdaptivePMMH(mcmc.PMMH):
    """Adaptive Particle Marginal Metropolis Hastings.

    PMMH is class of Metropolis samplers where the intractable likelihood of
    the considered state-space model is replaced by an estimate obtained from
    a particle filter. The proposal of MH sampler evolves throughout the chain
    following a scheme expounds in Knape et al. (2012).
    """

    def __init__(self, *args, **kwargs):
        assert all([kwarg in kwargs and type(kwargs[kwarg]) == int for kwarg in
                    ['m1', 'm2', 'update_interv']]), 'Please name the arguments m1, m2 and update_interv (integers)'
        assert all([kwarg in kwargs and type(kwargs[kwarg]) == float for kwarg in
                    ['w01', 'w02', 'w1', 'k0', 'k1']]), 'Please name the arguments w01, w02, w1, k0 and k1 (float)'
        m1 = kwargs.pop('m1')  # end of first adaptive phase
        m2 = kwargs.pop('m2')  # end of second phase
        update_interv = kwargs.pop('update_interv')  # at which rate should we update the gaussian mixture in the second phase
        w01 = kwargs.pop('w01')  # Weigth for iid RW-step proposal. (0.4 in the paper)
        w02 = kwargs.pop('w02')  # Weight for RW-step proposal. (0.5 in the paper)
        w1 = kwargs.pop('w1')  # Weight for mixture part without inflated variance (0.8 in the paper)
        k0 = kwargs.pop('k0')  # Variance inflation factor (phase 1)
        k1 = kwargs.pop('k1')  # Variance inflation factor (phase 2)

        if 'EM' in kwargs:  # EM algo for Gaussian Mixture (KMeans as in the article otherwise)
            self.EM = kwargs.pop('EM')
        else:
            self.EM = False

        super(AdaptivePMMH, self).__init__(*args, **kwargs)

        self.m1 = m1
        self.m2 = m2
        self.sequence = self.m1 + np.linspace(start=1,
                                              stop=self.m2 - self.m1,
                                              num=(self.m2 - self.m1) // update_interv)
        # iterations at which to update proposal during independence phase.
        self.w01 = w01
        self.w02 = w02
        self.w1 = w1
        self.k0 = k0
        self.k1 = k1

        assert w01 + w02 <= 1 and w1 <= 1 and w01 >= 0 and w02 >= 0 and w01 >= 0, 'Wrong weights given'
        assert m1 < m2 and 0 < m1 and m2 < kwargs['niter'], 'Wrong values for the schedule of the adaptive scheme'

        self.adaptive = True  # the adaptive scheme is always used with this class
        self.dicMixt = None
        self.PropGenLP = 0  # for the accept/reject part, the proposal likelihood
        self.GenLP = 0

        self.evidence = 0

        self.null_mean = np.zeros(self.chain.dim)

    def propDensity(self):
        """
        Returns
        -------
        A vector proposition for the MH update
        """
        if self.dicMixt is None:
            return stats.norm.rvs(size=self.chain.dim, scale=0.1)  # for the first proposals

        else:
            return self.mixtureRandom(self.dicMixt)

    # clustering in order to return a mixture (with the corresponding BIC)
    def clusteringMixture(self, p, comp_count=1):
        """
        Parameters
        ----------
        p: array
            Data to fit the Gaussian Mixture
        comp_count: int (default=None)
            number of components in the Gaussian Mixture model

        Returns
        -------
        Information concerning a potential mixture of Gaussians (Dictionary)
        """
        d = p.shape[1]  # dimension of observations
        n = p.shape[0]  # sample size

        mean_c = [np.zeros((d, 1)) for _ in range(comp_count)]  # cluster means
        var_c = [np.zeros((d, d)) for _ in range(comp_count)]  # cluster variances
        likelihood = np.zeros((comp_count, n))  # for BIC computation
        weights = []

        if not self.EM:
            # K-Means clustering
            cholesky = np.linalg.cholesky(np.cov(p, rowvar=False))
            choleskyinv = np.linalg.inv(cholesky)
            kmeans = cluster.KMeans(n_clusters=comp_count, n_jobs=-1).fit(
                    np.dot(p, np.transpose(choleskyinv))
                )
            labels = kmeans.labels_

            for i in range(comp_count):
                mean_c[i] = np.apply_along_axis(func1d=np.mean, axis=0, arr=p[labels == i, :])
                var_c[i] = np.cov(p[labels == i, :], rowvar=False)
                weights.append(np.sum(labels == i) / n)
                likelihood[i, :] = np.sum(labels == i) / n * stats.multivariate_normal.pdf(x=p,
                                                                                           mean=mean_c[i],
                                                                                           cov=var_c[i],
                                                                                           allow_singular=True)

        else:
            GM = mixture.GaussianMixture(n_components=comp_count, covariance_type='full', max_iter=1000)
            GM.fit(p)

            for i in range(comp_count):
                mean_c[i] = GM.means_[i]
                var_c[i] = GM.covariances_[i]
                weights.append(GM.weights_[i])
                likelihood[i, :] = GM.weights_[i]* stats.multivariate_normal.pdf(x=p,
                                                                                 mean=mean_c[i],
                                                                                 cov=var_c[i],
                                                                                 allow_singular=True)


        loglikelihood = np.sum(np.log(np.sum(likelihood, axis=0)))
        BIC = -2 * loglikelihood + (comp_count * (d * (d + 1) / 2 + d + 1)) * np.log(n)

        # Gaussian Mixture density
        if comp_count == 1:
            density = {"mean": mean_c[0], "variance": var_c[0]}
        else:
            composition = []
            for i in range(comp_count):
                composition.append({"mean": mean_c[i], "variance": var_c[i]})
            density = {"weights": weights, "composition": composition}

        return ({"density": density, "BIC": BIC})

    # Selection of the number of components for the mixture and outputs the corresponding density
    def proposalMixture(self, p):
        """
        Parameters
        ----------
        p: array
            Data to fit the Gaussian Mixtures
        Returns
        -------
        Information relative to the mixture of Gaussians (Dictionary)
        """

        BIC = np.array([np.nan for _ in range(6)])
        densitylist = np.array([{} for _ in range(6)])
        for i in range(1, 7):
            c = self.clusteringMixture(p, i)
            BIC[i - 1] = c["BIC"]
            densitylist[i - 1] = c["density"]
        index = np.where(BIC == np.min(BIC))[0][0]
        density = densitylist[index]  # selected density by BIC minimization

        density_ = density.copy()  # density with inflated variance
        if index == 0:
            density_["variance"] = self.k1 * density_["variance"]
            return (
                {"weights": [self.w1, 1 - self.w1], "composition": [density, density_]}
            )
        else:
            for component in range(index + 1):
                density_["composition"][component]["variance"] = self.k1 * density_["composition"][component][
                    "variance"]

        return ({"weights": [self.w1, 1 - self.w1], "composition": [density, density_]})

    # density of a Gaussian Mixture
    def mixtureDensity(self, x, mixture):
        """
        Parameters
        ----------
        x: array
           point whose density we want to compute
        mixture: dictionary
           A dictionary with informations necessary to the computation of the density (means, variances, weights..)

        Returns
        -------
        A float equal to the requested density
        """

        if "composition" not in mixture.keys():
            density = stats.multivariate_normal.pdf(x=x,
                                                    mean=mixture["mean"].ravel(),
                                                    cov=mixture["variance"],
                                                    allow_singular=True)
        else:
            density = sum(
                [
                    mixture["weights"][i] * self.mixtureDensity(x, mixture["composition"][i])
                    for i in range(len(mixture["weights"]))
                ]
            )
        return density

    # random number generation from a mixture
    def mixtureRandom(self, mixture):
        if "composition" not in mixture.keys():
            random = stats.multivariate_normal.rvs(
                mean=mixture["mean"].ravel(),
                cov=mixture["variance"],
                size=1)
        else:
            choice = np.int(np.random.choice(len(mixture["weights"]),
                                             size=1, replace=True, p=mixture["weights"]))
            random = self.mixtureRandom(mixture["composition"][choice])
        return random

    def step(self, n):
        d = self.chain.dim
        if n <= self.m1:  # first phase of the adaptation: Random-walk
            if self.nacc < 2 * self.chain.dim:
                z = self.propDensity()
                self.prop.arr[0] = self.chain.arr[n-1]+z
            else:
                z = stats.norm.rvs(size=d)
                choice = np.int(np.random.choice(3, size=1,
                                                 p=[self.w01, self.w02, 1-self.w01-self.w02]))
                factor = [np.sqrt(0.1)*np.eye(d), self.L, np.sqrt(self.k0)*self.L][choice]
                self.prop.arr[0] = self.chain.arr[n - 1] + np.dot(factor, z)
            self.compute_post()
            lp_acc = self.prop.lpost[0] - self.chain.lpost[n - 1]
            if np.log(stats.uniform.rvs()) < lp_acc:  # accept
                self.chain.copyto_at(n, self.prop, 0)
                self.nacc += 1
            else:  # reject
                self.chain.copyto_at(n, self.chain, n - 1)

            self.cov_tracker.update(self.chain.arr[n])
            self.L = self.scale * self.cov_tracker.L

        else:  # Independance phase
            if n in self.sequence:
                self.dicMixt = self.proposalMixture(self.chain.arr[:n - 1])
                self.GenLP = np.log(self.mixtureDensity(self.chain.arr[n - 1], self.dicMixt))

            self.prop.arr[0] = self.propDensity()
            self.compute_post()
            self.PropGenLP = np.log(self.mixtureDensity(self.prop.arr[0], self.dicMixt))

            if n > self.m2:
                self.evidence += np.exp(self.prop.lpost[0] - self.PropGenLP)

            lp_acc = self.prop.lpost[0] - self.chain.lpost[n - 1] - (self.PropGenLP - self.GenLP)
            if np.log(stats.uniform.rvs()) < lp_acc:  # accept
                self.chain.copyto_at(n, self.prop, 0)
                self.nacc += 1
                self.GenLP = self.PropGenLP
            else:  # reject
                self.chain.copyto_at(n, self.chain, n - 1)

    # def step_(self, n):
    #     if n <= self.m1:  # first phase of the adaptation: Random-walk
    #         if self.nacc >= 2 * self.chain.dim:
    #             self.dicMixt = {
    #                 'weights': [self.w01, self.w02, 1 - self.w01 - self.w02],
    #                 'composition': [
    #                     {"mean": self.null_mean, "variance": 0.1 * np.diag(np.repeat(1, self.chain.dim))},
    #                     {"mean": self.null_mean, "variance": (2.38**2 / self.chain.dim) * self.L},
    #                     {"mean": self.null_mean, "variance": (self.k0 * 2.38**2 / self.chain.dim) * self.L}
    #                 ]
    #             }
    #
    #         z = self.propDensity()
    #         self.prop.arr[0] = self.chain.arr[n - 1] + z
    #         self.compute_post()
    #         lp_acc = self.prop.lpost[0] - self.chain.lpost[n - 1]
    #         if np.log(stats.uniform.rvs()) < lp_acc:  # accept
    #             self.chain.copyto_at(n, self.prop, 0)
    #             self.nacc += 1
    #         else:  # reject
    #             self.chain.copyto_at(n, self.chain, n - 1)
    #
    #         self.cov_tracker.update(self.chain.arr[n])
    #         # self.L = self.scale * self.cov_tracker.L
    #         self.L = np.cov(self.chain.arr[:n], rowvar=False)
    #
    #     else:  # Independance phase
    #         if n in self.sequence:
    #             self.dicMixt = self.proposalMixture(self.chain.arr[:n - 1])
    #             self.GenLP = np.log(self.mixtureDensity(self.chain.arr[n - 1], self.dicMixt))
    #
    #         self.prop.arr[0] = self.propDensity()
    #         self.compute_post()
    #         self.PropGenLP = np.log(self.mixtureDensity(self.prop.arr[0], self.dicMixt))
    #
    #         if n > self.m2:
    #             self.evidence += np.exp(self.prop.lpost[0] - self.PropGenLP)
    #
    #         lp_acc = self.prop.lpost[0] - self.chain.lpost[n - 1] - (self.PropGenLP - self.GenLP)
    #         if np.log(stats.uniform.rvs()) < lp_acc:  # accept
    #             self.chain.copyto_at(n, self.prop, 0)
    #             self.nacc += 1
    #             self.GenLP = self.PropGenLP
    #         else:  # reject
    #             self.chain.copyto_at(n, self.chain, n - 1)

    def run(self):
        super(AdaptivePMMH, self).run()
        self.evidence = np.log(self.evidence)-np.log(self.niter - self.m2)