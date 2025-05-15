from itertools import chain

import numpy as np

from pybigue.models import angular_distance


def normalized_reg_autocovariance(x, lag):
    """Returns autocovariance normalized with the 0-lag autocovariance."""
    return reg_autocovariance(x, lag)/reg_autocovariance(x, 0)


def normalized_circ_autocovariance(x, lag):
    """Returns circular autocovariance normalized with the 0-lag circular autocovariance."""
    return circ_autocovariance(x, lag)/circ_autocovariance(x, 0)


def circular_average(thetas):
    """Returns circular average
    :math:`\\arg \\left( \\sum_{j=1}^S \\exp{i\\phi_j} \\right)`.
    """
    # Returns 0 instead of undefined when the average complex number is 0.
    return np.angle(np.average(np.exp(1j*thetas)))


def get_max_lag(T):
    """Returns max lag threshold when computing effective sample size based on the total sample
    size."""
    return T//100

def circ_autocorrelation(theta_sample, lag):
    """Returns the autocorrelation adapted for sample of a random variable on the circle."""
    if lag == 0:
        return 1
    x = theta_sample[:-lag]
    y = theta_sample[lag:]
    sin_xdiff = np.sin(x-circular_average(x))
    sin_ydiff = np.sin(y-circular_average(y))
    num = np.sum(sin_xdiff*sin_ydiff)
    denom = np.sqrt(np.sum(sin_xdiff**2)) * np.sqrt(np.sum(sin_ydiff**2))
    return num/denom


def reg_autocorrelation(param, lag):
    """Returns sample autocorrelation."""
    x = param[:-lag]
    y = param[lag:]
    xdiff = x-np.average(x)
    ydiff = y-np.average(y)
    denom = np.sqrt(np.sum(xdiff**2)) * np.sqrt(np.sum(ydiff**2))
    return np.sum(xdiff*ydiff)/denom


def circ_autocovariance(theta_sample, lag):
    """Returns sample autocovariance adapted for a random variable on the circle."""
    average = circular_average(theta_sample)
    x = theta_sample[lag:]
    if lag == 0:
        y = x
    else:
        y = theta_sample[:-lag]
    return np.sum(np.sin(x-average)*np.sin(y-average))


def reg_autocovariance(sample, lag):
    """Returns sample autocovariance."""
    average = np.average(sample)
    x = sample[lag:]
    if lag == 0:
        y = x
    else:
        y = sample[:-lag]
    return np.sum((x-average)*(y-average))


def circ_seff_rhat(theta_samples):
    """Combined effective sample size of Markov chains adapted for a random
    variable on the circle. The chains are split in two to compute the
    autocovariance."""
    split_chains = list(chain.from_iterable([[ts[:len(ts)//2], ts[len(ts)//2:]] for ts in theta_samples]))
    m = len(split_chains)
    n = len(split_chains[0])

    averages = np.array([circular_average(chain) for chain in split_chains])
    global_average = circular_average(averages)
    chain_var = [1/(n-1)*np.sum(angular_distance(chain, averages[j])**2) for j, chain in enumerate(split_chains)]
    W = np.average(chain_var)
    B = n/(m-1)*sum(angular_distance(average, global_average)**2 for average in averages)
    var_p = (n-1)/n * W + B / n

    max_lag = n//50
    rho = [1-((W - np.average(
                chain_var * np.array([normalized_circ_autocovariance(chain, t) for chain in split_chains])))
              /var_p)
           for t in range(1, max_lag+1)]

    rhat = np.sqrt(1/n*(n-1 + B/W))
    seff = n*m/(1+2*sum(rho))

    return seff, rhat


def reg_seff_rhat(theta_samples):
    """Combined effective sample size of Markov chains. The chains are split in
    two to compute the autocovariance."""
    split_chains = list(chain.from_iterable([[ts[:len(ts)//2], ts[len(ts)//2:]] for ts in theta_samples]))
    m = len(split_chains)
    n = len(split_chains[0])

    chain_var = np.var(split_chains, axis=1, ddof=1)
    W = np.average(chain_var)
    B = n*(np.var(np.average(split_chains, axis=1), ddof=1))
    var_p = (n-1)/n * W + B / n

    max_lag = n//50
    rho = [1-((W - np.average(
                chain_var * np.array([normalized_reg_autocovariance(chain, t) for chain in split_chains])))
              /var_p)
           for t in range(1, max_lag+1)]

    rhat = np.sqrt(1/n*(n-1 + B/W))
    seff = n*m/(1+2*sum(rho))

    return seff, rhat


def circ_neff(theta_sample):
    """Effective sample size of a random variable on the circle."""
    T = len(theta_sample)
    m = get_max_lag(T)
    a_0 = circ_autocovariance(theta_sample, 0)
    return T/(1+2/a_0*sum(circ_autocovariance(theta_sample, t) for t in range(1, m+1)))


def reg_neff(sample):
    """Effective sample size."""
    T = len(sample)
    m = get_max_lag(T)
    a_0 = reg_autocovariance(sample, 0)
    return T/(1+2/a_0*sum(reg_autocovariance(sample, t) for t in range(1, m+1)))
