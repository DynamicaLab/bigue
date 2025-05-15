import numpy as np
from numba import njit
from scipy import stats


@njit
def align_theta(theta, vertex_at_zero, vertex_upper_half):
    """Rotates and flips `theta` such that `vertex_at_zero` is at 0 and
    `vertex_upper_half` is in [0, pi]."""
    fixed_angle = theta[vertex_at_zero]
    adjusted = np.copy(theta)-fixed_angle
    adjusted[adjusted<-np.pi] = adjusted[adjusted<-np.pi] + 2*np.pi
    adjusted[adjusted>np.pi] = adjusted[adjusted>np.pi] - 2*np.pi

    if adjusted[vertex_upper_half] < 0:
        return -adjusted
    return adjusted


@njit
def log1pexp(x):
    """Numerically stable function computing log(1+exp(x)) obtained from from
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf.
    """
    if x <= -37:
        return np.exp(x)
    elif x <= 18:
        return np.log1p(np.exp(x))
    elif x <= 33.3:
        return x + np.exp(-x)
    return x


def angle_modulo(theta):
    """Change `theta` values such that they are in [-pi, pi]."""
    theta = np.asarray(theta)
    pi_to_add = np.ceil(abs(np.min(theta) / (2 * np.pi)))
    return np.mod(theta + (2 * pi_to_add + 1) * np.pi, 2 * np.pi) - np.pi


def sample_truncated_normal(average, std, size,
                            lower_bound=None,
                            upper_bound=None):
    """Samples truncated normal distribution with rejection sampling. If some
    bound is `None`, it is set to infinity (no bounds).

    Code is adapted from https://stackoverflow.com/questions/47933019/how-to-properly-sample-truncated-distributions.
    """
    if lower_bound is not None and upper_bound is not None:
        condition = lambda x: (x >= lower_bound) & (x <= upper_bound)
    elif lower_bound is not None:
        condition = lambda x: (x >= lower_bound)
    elif upper_bound is not None:
        condition = lambda x: (x <= upper_bound)
    else:
        return stats.norm.rvs(average, std, size=size)

    proposal = stats.halfnorm if lower_bound == 0 else stats.norm

    samples = np.zeros((0, ))  # empty for now
    while samples.shape[0] < size:
        s = proposal.rvs(average, std, size=size)
        accepted = s[condition(s)]
        samples = np.concatenate((samples, accepted), axis=0)
    if size == 1:
        return samples[0]
    return samples[:size]


def sample_truncated_pareto(lower_bound, upper_bound, exponent, size):
    """Rejection sampling algorithm to sample truncated pareto distribution."""
    samples = np.zeros((0,))
    while samples.shape[0] < size:
        s = lower_bound * stats.pareto.rvs(exponent-1, size=size)
        accepted = s[s <= upper_bound]
        samples = np.concatenate((samples, accepted), axis=0)
    return samples[:size]


def truncated_normal_lpdf(x, average, std, lower_bound=None, upper_bound=None):
    """Log-density of the truncated normal distribution at `x`."""
    lower_mass = 0 if lower_bound is None else stats.norm.cdf(lower_bound)
    upper_mass = 0 if upper_bound is None else 1-stats.norm.cdf(upper_bound)
    params = {"loc": average, "scale": std}
    return stats.norm.logpdf(x, **params) - np.log(1-lower_mass-upper_mass)


def gen_pareto_lpdf(power, min_val):
    """Returns function that computes the log density of a power-law distribution."""
    return lambda x: power*np.log(min_val)-np.log(np.asarray(x))


def gen_normal_lpdf(average, std, min_val=None):
    """Returns function that computes the log density of a truncated normal distribution."""
    return lambda x: truncated_normal_lpdf(x, average=average, std=std, lower_bound=min_val)


def gen_cauchy_lpdf(median, gamma):
    """Returns function that computes the log density of a cauchy distribution."""
    return lambda x: -np.log1p(((np.asarray(x)-median)/gamma)**2)


def sample_uniform(lower_bound, upper_bound, size):
    """Samples uniform distribution."""
    return (upper_bound-lower_bound)*np.random.rand(size) + lower_bound
