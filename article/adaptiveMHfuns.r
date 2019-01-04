## This file contains functions that are used for estimating and handling the normal mixture proposals
## Normal mixtures are coded as recursive lists where first element, named w, is a list of weigths
## for each of a number of components where the components are themselves mixtures. The second element
## is either i) if length(w)>1 another mixture or ii) if length(w)==1 a normal distribution coded as
## a list containing a mean vector and a variance matrix.

## Computes a normal mixture consisting of two parts, one mixture that approximates the posterior
## according to the MCMC trace and a clone of that mixture with inflated variance. The number of
## components of the mixtures is determined by BIC.
##
## Input: ptrace - history of MCMC
##        w - weight for approximating mixture component,
##            a weight of 1-w is devoted to variance inflated component.
##        k - inflation factor for variance inflated mixture
##        maxComp - maximum number of components for mixture approximation
## Output: a normal mixture
getPropDensity = function(ptrace, k, w, maxComp) {
    BIC = rep(0, maxComp)
    denslist = list()
    for (i in 1:maxComp) {
        ml = tryCatch(mixtlikapproximation(ptrace, i), error = function(e) list(density = NULL, BIC = Inf))
        j = 1
        while (j < 3 & is.nan(ml$BIC)) {
            ml = tryCatch(mixtlikapproximation(ptrace, i), error = function(e) list(density = NULL, BIC = Inf))
            j = j + 1
        }
        BIC[i] = ml$BIC
        denslist = append(denslist, list(ml$density))
    }
    nComp = which(BIC == min(BIC))
    density = denslist[[nComp]]
    extdensity = density
    if (nComp == 1) {
        extdensity$V = k * extdensity$V
    } else {
        for (i in 1:nComp) {
            extdensity$comp[[i]]$V = k * extdensity$comp[[i]]$V
        }
    }
    list(w = c(w, 1 - w), comp = list(density, extdensity))
}

# Computes a mixture for a given number of components using kmeans clustering.
## Input: ptrace - history of MCMC
##        maxComp - maximum number of components for mixture approximation
## Output: a list containing a normal mixture and the BIC of that mixture
##         (BIC is computed assuming that kmeans correctly classifies parameters).
mixtlikapproximation = function(ptrace, nComp) {
    d = dim(ptrace)[1]
    n = dim(ptrace)[2]
    # Transform to equal variances before applying kmeans
    chL = t(chol(var(t(ptrace))))
    chLinv = solve(chL)
    km = kmeans(t(chLinv %*% ptrace), centers = nComp, nstart = 20)
    pw = tabulate(km$cluster)/n
    m = matrix(rep(0, nComp * d), c(d, nComp))
    v = matrix(rep(0, nComp * d^2), c(d, nComp * d))
    lik = matrix(rep(0, n * nComp), c(nComp, n))
    for (i in 1:nComp) {
        m[, i] = apply(ptrace[, km$cluster == i], 1, mean)
        v[, (d * (i - 1) + 1):(d * i)] = var(t(ptrace[, km$cluster == i]))
        lik[i, ] = pw[i] * dmvnorm(t(ptrace), mean = m[, i], sigma = v[, (d * (i - 1) + 1):(d * i)])
    }
    llik = sum(log(apply(lik, 2, sum)))
    BIC = -2 * llik + (nComp * (d * (d + 1)/2 + d + 1)) * log(n)
    if (nComp == 1) {
        density = list(m = m, V = v)
    } else {
        comp = list()
        for (i in 1:nComp) {
            comp = append(comp, list(list(m = m[, i], V = v[, (d * (i - 1) + 1):(d * i)])))
        }
        density = list(w = pw, comp = comp)
    }
    list(density = density, BIC = BIC)
}

# Returns the density of mixture at x
mixtureDensity = function(x, mixture, log = FALSE) {
    ncomp = length(mixture$w)
    if (ncomp == 0) {
        d = dmvnorm(x, mean = mixture$m, sigma = mixture$V)
    } else {
        d = 0
        for (i in 1:ncomp) {
            d = d + mixture$w[i] * mixtureDensity(x, mixture = mixture$comp[[i]], log = FALSE)
        }
    }
    if (log) 
        d = log(d)
    d
}

# Draws a random number from mixture
mixtureSim = function(mixture) {
    ncomp = length(mixture$w)
    if (ncomp == 0) {
        r = rmvnorm(n = 1, mean = mixture$m, sigma = mixture$V, method = "chol")
    } else {
        c = sample(ncomp, 1, prob = mixture$w)
        r = mixtureSim(mixture = mixture$comp[[c]])
    }
    r
}
 
