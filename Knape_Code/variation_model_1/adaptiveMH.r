## This script estimates a random walk model with drift for Kangaroo population
## dynamics on the log-scale using particle filtering Metropolis Hastings with an initial
## adaptive phase.
##
## Written by Jonas Knape (jknape@berkeley.edu)

rm(list = ls())

require(mvtnorm)
source("adaptiveMHfuns.r")  # Contains support functions for adaptive Metropolis Hastings
source("PF.r")  # Defines particle filter and priors for the model

##################################################################
# Data
##################################################################

# From Caughley et al. 1987.
y = matrix(c(267, 333, 159, 145, 340, 463, 305, 329, 575, 227, 532, 769, 526, 565, 466, 494, 440, 858, 599, 298, 529, 912, 703, 402, 
    669, 796, 483, 700, 418, 979, 757, 755, 517, 710, 240, 490, 497, 250, 271, 303, 386, 326, 144, 145, 138, 413, 531, 331, 329, 529, 318, 449, 
    852, 332, 742, 479, 620, 531, 751, 442, 824, 660, 834, 955, 453, 953, 808, 975, 627, 851, 721, 1112, 731, 748, 675, 272, 292, 389, 323, 
    272, 248, 290, 1973.497, 1973.75, 1974.163, 1974.413, 1974.665, 1975.002, 1975.245, 1975.497, 1975.75, 1976.078, 1976.33, 1976.582, 1976.917, 
    1977.245, 1977.497, 1977.665, 1978.002, 1978.33, 1978.582, 1978.832, 1979.078, 1979.582, 1979.832, 1980.163, 1980.497, 1980.75, 1980.917, 
    1981.163, 1981.497, 1981.665, 1981.917, 1982.163, 1982.413, 1982.665, 1982.917, 1983.163, 1983.413, 1983.665, 1983.917, 1984.163, 1984.413), 
    c(3, 41), byrow = T)

d = 4 # Number of parameters to estimate
# M = 50000  # Total length of MCMC chain
# N = 4000  # Number of particles
M = 10000
N = 4000
# r, sigma, tau, b
# inits = c(2, 2, 0.5, 0.05)
inits = c(1, 1, 0.1, 0.01)  # Initial values
particleFilterLL(y, inits, N)$LL

##################################################################
# Adaptation parameters
##################################################################

## Initial RW phase

trs = 2000  # Length of initial RW
w01 = 0.4  # Weigth for iid RW-step proposal.
w02 = 0.4  # Weight for RW-step proposal.
k0 = 25  # Variance inflation factor.

## Independence phase

# Iterations at which to update proposal during independence phase.
updateSeq = trs + c(seq(1, 4000 - trs - 1, by = 50))

w1 = 0.8  # Weight for mixture part without inflated variance
k1 = 5  # Variance inflation factor
maxComp = 4  # Maximum number of components for normal mixture


##################################################################
# Initialize variables
##################################################################

pars = matrix(rep(0, M * d), c(d, M))  # Matrix for holding the trace of the MCMC
propPars = matrix(rep(0, M * d), c(d, M))  # Matrix for holding proposed parameter values
states = matrix(rep(0, M * length(y[1, ])), c(length(y[1, ])), M)  # Matrix for states

LL = rep(0, M)  # Log likelihood of current parameter values
priorLP = rep(0, M)  # Prior log probability of current parameter values
genLP = rep(0, M)  # Log probability of proposing current parameter values
propLL = rep(0, M)  # Log likelihood of proposed parameter values
propPriorLP = rep(0, M)  # Prior log probability of proposed parameter values
propGenLP = rep(0, M)  # Log probability of generating proposal

accepted = rep(0, M)  # Indicator for whether proposed values are accepted

pfOutput = NULL
propDen = NULL  # Proposal density

# Set MCMC starting values
pars[, 1] = inits
pfOutput = particleFilterLL(y, pars[, 1], N)
states[, 1] = pfOutput$rN
LL[1] = pfOutput$LL
priorLP[1] = prior(pars[, 1])

pb = txtProgressBar(title = "Progress", style = 1, min = 1, max = M)

##################################################################
# Adaptive Metropolis Hastings
##################################################################

for (i in 2:M) {
    # Compute proposal density
    if (i <= trs) {
        if (sum(accepted[1:i]) <= 2 * d) {
          # propDen = list(m = pars[, i - 1], V = 0.1 * diag(rep(1, d))/d)
          # propDen = list(m = pars[, i - 1], V = 0.1 * diag(c(1, 1, 0.1, 0.01))/d)
          propDen = list(m = pars[, i - 1], V = 0.1 * diag(c(1, 1, 0.1, 0.01))/d)
        } else {
            propDen = list(w = c(w01, w02, 1 - w01 - w02), comp = list(list(m = pars[, i - 1], V = 0.1 * diag(c(1, 1, 0.1, 0.01))/d), list(m = pars[, 
                i - 1], V = 2.38^2 * var(t(pars[, 1:(i - 1)]))/d), list(m = pars[, i - 1], V = k0 * 2.38^2 * var(t(pars[, 1:(i - 1)]))/d)))
            # propDen = list(w = c(w01, w02, 1 - w01 - w02), comp = list(list(m = pars[, i - 1], V = 0.1 * diag(rep(1, d))/d), list(m = pars[, 
            #     i - 1], V = 2.38^2 * var(t(pars[, 1:(i - 1)]))/d), list(m = pars[, i - 1], V = k0 * 2.38^2 * var(t(pars[, 1:(i - 1)]))/d)))
        }
    } else if (is.element(i, updateSeq)) {
        # Update proposal at predetermined iterations
        propDen = getPropDensity(pars[, 1:(i - 1)], k1, w1, maxComp)
    }
    # Generate proposal
    propPars[, i] = mixtureSim(propDen)
    pfOutput = particleFilterLL(y, propPars[, i], N)
    propLL[i] = pfOutput$LL
    propPriorLP[i] = prior(propPars[, i])
    propGenLP[i] = mixtureDensity(propPars[, i], propDen, log = TRUE)
    # Accept or reject proposal
    if (log(runif(1, 0, 1)) < propLL[i] + propPriorLP[i] - LL[i - 1] - priorLP[i - 1] + (i > trs) * (genLP[i - 1] - propGenLP[i])) {
        states[, i] = pfOutput$rN
        pars[, i] = propPars[, i]
        LL[i] = propLL[i]
        priorLP[i] = propPriorLP[i]
        genLP[i] = propGenLP[i]
        accepted[i] = 1
    } else {
        states[, i] = states[, i - 1]
        pars[, i] = pars[, i - 1]
        LL[i] = LL[i - 1]
        priorLP[i] = priorLP[i - 1]
        genLP[i] = genLP[i - 1]
    }
    if (i%%floor(M/1000) == 0) {
        Sys.sleep(0.01)
        setTxtProgressBar(pb, i)
        par(mfrow = c(3, 2))
        plot(pars[1, 1:i], xlab = "iteration", ylab = "r", type = "l")
        plot(pars[2, 1:i], xlab = "iteration", ylab = expression(sigma), type = "l")
        plot(pars[3, 1:i], xlab = "iteration", ylab = expression(tau), type = "l")
        plot(pars[4, 1:i], xlab = "iteration", ylab = "b", type = "l")
        plot(LL[1:i], xlab = "iteration", ylab = "log-likelihood", type = "l")
        # plot(states[25, 1:i], xlab = "iteration", ylab = expression(N[25]), type = "l")
    }
}

close(pb)

##################################################################
# Analysis
##################################################################

# Traceplots
par(mfrow = c(2, 2))
plot(pars[1, ], xlab = "iteration", ylab = "r", type = "l")
plot(pars[2, ], xlab = "iteration", ylab = expression(sigma), type = "l")
plot(pars[3, ], xlab = "iteration", ylab = expression(tau), type = "l")
plot(pars[4, ], xlab = "iteration", ylab = "b", type = "l")
# plot(states[25, ], xlab = "iteration", ylab = expression(N[25]), type = "l")

save(pars, file = "model 1/pars_model1")

# Discard burnin iterations
I = 4000:M

# Parameter estimates
par(mfcol = c(2, 2))
hist(pars[1, I], xlab = "r", ylab = "frequency", breaks = 500)
hist(pars[2, I], xlab = expression(sigma), ylab = "frequency", breaks = 500)
hist(pars[3, I], xlab = expression(tau), ylab = "frequency", breaks = 500)
hist(pars[4, I], xlab = "b", ylab = "frequency", breaks = 500)
# hist(states[25, I], xlab = expression(N[25]), ylab = "frequency", breaks = 500)

# State estimates
par(mfcol = c(1, 1))
plot(y[3, ], y[1, ], col = "red", xlab = "year", ylab = "#kangaroos")
points(y[3, ], y[2, ], col = "blue")
points(y[3, ], apply(states[, I], 1, mean), type = "l")

# Marginal density
plot(log(cumsum(exp(propLL[I] + propPriorLP[I] - propGenLP[I]))/(1:length(I))), xlab = "iteration", ylab = "marginal density estimate", 
    type = "l", col = "red")
margDens = log(mean(exp(propLL[I] + propPriorLP[I] - propGenLP[I])))
