# Kangaroos #
Particle Filters with Markov Chain Monte Carlo

This project was developed following a course from N.Chopin at ENSAE ParisTech. It aims at reproducing the results from [1]. In this paper, Knape et al. attempted to estimate population state-space models using a Particle filter Metropolis Hastings algorithm. They used an adaptive scheme to update the proposal of the Metropolis Hastings algorithm and improve the mixing.

The code heavily relies on the package [particles](https://github.com/nchopin/particles) for the implementation of our PMMH algorithm.

[ 1 ] Fitting complex population models by combining particle filters with Markov chain Monte Carlo - Jonas Knape, Perry de Valpine

# Implementation #

The folder *ParticleMCMC* contains our implementation of the different functions used to model the state-space models as well as those used to perform the adaptive PMMH algorithm. In the sub-directory *Experiments*, you will find the notebooks that we used to produce our different results.

# Original work from Knape et al. #

The original R code developed by the authors of article under scrutiny is also available here, in case you would like to reproduce their results.

# Bibliography #

If you would seek some resources linked to this project, we have gathered some articles that we considered useful. You can find them in the *articles* folder.
