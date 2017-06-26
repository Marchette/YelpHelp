"""Probabilistic programming function designed to find change points in Yelp reviews: mcmc_changepoint"""

import numpy as np
import pymc3 as pm
from pymc3 import Model, DiscreteUniform, Exponential, Poisson, Normal, HalfNormal, NUTS, Metropolis, sample, traceplot, find_MAP
from pymc3.math import switch
import matplotlib.pyplot as plt
import scipy

def mcmc_changepoint(dates, ratings, mcmc_iter=1000, discrete=0, plot_result=1):
    """This function models Yelp reviews as coming from two normal distributions
    with a switch point somewhere between them. When left of the switch point then
    reviews are drawn from the first normal distribution. To the right of the
    switch point reviews are drawn from the second normal distribution. Normal
    distributions are used if the reviews have been normalized to the user's
    average rating; otherwise if analyzing in terms of 1-5 stars set discrete=1
    and the function will do the same estimation on Poisson distributions. This
    function then finds the most likely distribution for where the switchpoint is
    and the most likely parameters for the two generator distributions by using
    Metropolis-Hastings sampling and Hamiltonian Monte Carlo."""

    # dates: Array of dates when the reviews were posted
    # ratings: Array of the ratings given by each review
    # mcmc_iter: How many iterations of the MCMC to run?
    # discrete: Should I use Normal or Poisson distributions to model the ratings?
    # (i.e. are the user-averaged or 1-5 stars)
    # plot_result: Should the function output a plot?

    number_of_ratings = np.arange(0, len(ratings))

    if discrete == 0:
        with Model() as switch_model:
            switchpoint = DiscreteUniform('switchpoint', lower=0, upper=len(dates))

            before_intensity = Normal('before_intensity', mu=0, sd=1)
            after_intensity = Normal('after_intensity', mu=0, sd=1)

            intensity = switch(switchpoint >= number_of_ratings, before_intensity, after_intensity)
            sigma = HalfNormal('sigma', sd=1)

            rating = Normal('rating', mu=intensity, sd=sigma, observed=ratings)

    elif discrete == 1:
        with Model() as switch_model:
            switchpoint = DiscreteUniform('switchpoint', lower=0, upper=len(dates))

            before_intensity = Exponential('before_intensity', 1)
            after_intensity = Exponential('after_intensity', 1)

            intensity = switch(switchpoint >= number_of_ratings, before_intensity, after_intensity)

            rating = Poisson('rating', intensity, observed=ratings)

    with switch_model:
        trace = sample(mcmc_iter)

    if plot_result == 1:
        traceplot(trace)
        plt.show()

    switch_posterior = trace['switchpoint']
    N_MCs = switch_posterior.shape[0]

    before_intensity_posterior = trace['before_intensity']
    after_intensity_posterior = trace['after_intensity']

    expected_stars = np.zeros(len(ratings))
    for a_rating in number_of_ratings:
        where_switch = a_rating < switch_posterior
        expected_stars[a_rating] = (before_intensity_posterior[where_switch].sum() + after_intensity_posterior[
            ~where_switch].sum()) / N_MCs

    if plot_result == 1:
        plt.plot(dates, ratings, 'o')
        plt.plot(dates, expected_stars, 'b-')
        plt.show()

    # Return the mode and it's frequency / mcmc_iter
    b_mean, b_count = scipy.stats.mode(trace['before_intensity'])
    a_mean, a_count = scipy.stats.mode(trace['after_intensity'])
    modal_switch, count = scipy.stats.mode(trace['switchpoint'])
    sigma_est, sigma_count = scipy.stats.mode(trace['sigma'])
    differential = b_mean - a_mean
    return differential, modal_switch, expected_stars, sigma_est, switch_posterior