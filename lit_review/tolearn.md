# Stuff I need to learn

## multi-fidelity bayesian optimization

## bayesian neural network

## max-value entropy search

## hamiltonian monte-carlo

- [wiki](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo)
- a special case of the [metropolis-hastings algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm).

## mutual information

- [wiki](https://en.wikipedia.org/wiki/Mutual_information)
- quantifies the information that observing one random variable tells you about another random variable

## gamma distribution

- [wiki](https://en.wikipedia.org/wiki/Gamma_distribution)
- exponential and chi squared are special gamma distributions
- pdf: 
	$$ f(x) = \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{\alpha - 1} e^{- \beta x}$$
- mean is $\frac{\alpha}{\beta}$; variance is $\frac{\alpha}{\beta^2}$.
- $\Gamma(\cdot)$ is the gamma function
- gamma distribution is used as a conjugate prior over $\lambda$ in Poisson distribution.

## current hyperparameter optimization techniques

- [wiki](https://en.wikipedia.org/wiki/Hyperparameter_optimization)
- cross validation is used as proxy for population performance
- common techniques:
	- grid search
	- random search
	- bayesian optimization
	- gradient-based optimization
	- evolutionary optimization
	- population-based
	- early stopping-based

## physics informed neural networks (maybe)

## online learning latent dirichlet allocation (maybe)