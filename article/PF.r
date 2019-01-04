## Particle filter for the Random walk model with drift:
## y[1,t], y[2,t] ~ NegBin(mean=exp(x[t]),tau)
## x[t+1]~N(a+x[t],s^2 * delta(t)) where delta(t) is the time between sample t and t+1.
##
## Input: y data in a 3xT matrix where the first two rows contain the replicate observations
##        at eachh sampling occasion and the third row contains the time of sampling,
##        p=c(r, sigma, tau), N = number of partcles.
## Output: A list containing the estimate of the log likelihood (LL) and an
##         approximate draw from the filtering density.
particleFilterLL = function(y, p, N) {
    if (min(p)<0)
        return(list(LL = -Inf, rN = NULL))
    r = p[1]
    s = p[2]
    tau = p[3]
    n = dim(y)[2]  # Length of the time series
    x = matrix(rep(0, N * n), c(N, n))  # A matrix containing the particles at the log scale
    rx = rep(0, n)
    ind = matrix(rep(0, N * n), c(N, n))  # Index for the previous position of each particle
    P = rep(0, n)  # Conditional likelihood at each time step
    
    x[, 1] = rnorm(N, 5, 10)  # Draw particles at time 1 from the prior
    # logW is a vector of log-weights
    logW = dnegbin(y[1, 1], mu = exp(x[, 1]), tau = tau) + dnegbin(y[2, 1], mu = exp(x[, 1]), tau = tau)
    logWMax = max(logW)
    scaledWeights = exp(logW - logWMax)
    P[1] = mean(scaledWeights) * exp(logWMax)
    for (t in 2:n) {
        if (logWMax == -Inf) {
            return(list(LL = -Inf, rN = NULL))
        }
        # Resample particles
        ind[, t - 1] = sample(N, size = N, prob = scaledWeights, replace = TRUE)
        
        # Project particles forward
        x[, t] = x[ind[, t - 1], t - 1] + r * (y[3, t] - y[3, t - 1]) + s * sqrt(y[3, t] - y[3, t - 1]) * rnorm(N, 0, 1)
        
        # Compute new weights, likelihood contribution and effective sample size
        logW = dnegbin(y[1, t], mu = exp(x[, t]), tau = tau) + dnegbin(y[2, t], mu = exp(x[, t]), tau = tau)
        logWMax = max(logW)
        scaledWeights = exp(logW - logWMax)
        P[t] = mean(scaledWeights) * exp(logWMax)
    }
    
    # Traceback to generate simulated path
    ind[, n] = sample(N, size = N, prob = scaledWeights, replace = TRUE)
    itrace = ind[1, n]
    rx[n] = x[itrace, n]
    for (t in (n - 1):1) {
        itrace = ind[itrace, t]
        rx[t] = x[itrace, t]
    }
    list(LL = sum(log(P)), P = P, rN = exp(rx))
}

## Computes the log of the prior probabitlity for the parameters
##
## Input: p[1]=r, p[2]=sigma, p[3]=tau
## Output: log prior probability of the parameter vector
prior = function(p) {
    rmax = 10  # r~U(-rmax,rmax)
    smax = 10  # s~U(0,smax)
    taumax = 10  # tau~U(0,taumax)
    log((abs(p[1]) < rmax)/(2 * rmax)) + log((p[2] > 0 & p[2] < smax)/smax) + log((p[3] > 0 & p[3] < taumax)/taumax)
}

dnegbin = function(y, mu, tau) {
    Ld = lgamma(y + 1/tau) - lgamma(y + 1) - lgamma(1/tau) - (y + 1/tau) * log(1 + mu * tau) + y * log(mu * tau)
    Ld[is.nan(Ld)] = -Inf
    Ld
}
 
