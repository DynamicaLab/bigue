import warnings

import numpy as np
import numpy.typing as npt
from numba import njit
from scipy import optimize

from .embedding_info import EmbeddingParameters, EmbeddingsContainer
from .utils import angle_modulo, log1pexp


class S1Model:
    @staticmethod
    def get_edge_prob(embedding, average_degree, i, j):
        """Return probability that an edge connects vertices `i` and `j`."""
        theta, kappa, beta = embedding.theta, embedding.kappa, embedding.beta
        if theta is None:
            raise ValueError("Theta cannot be None in edge prob.")
        if kappa is None:
            raise ValueError("Kappa cannot be None in edge prob.")
        if beta is None:
            raise ValueError("Beta cannot be None in edge prob.")
        n = len(theta)

        R_div_mu = n * average_degree / (beta * np.sin(np.pi / beta))
        return 1 / (1 + (R_div_mu * angular_distance(theta[i], theta[j]) /
                         (kappa[i] * kappa[j]))**beta)

    @staticmethod
    @njit
    def loglikelihood(adjacency, average_degree, theta, kappa, beta):
        """Compute log-likelihood of the model from adjacency matrix."""
        total = 0

        n = len(theta)
        R_div_mu = n * average_degree / (beta * np.sin(np.pi / beta))
        for i in range(n - 1):
            for j in range(i + 1, n):
                chi = R_div_mu * angular_distance_1d(
                    theta[i], theta[j]) / kappa[i] / kappa[j]
                sign = 1 if adjacency[i][j] == 1 else -1
                total -= log1pexp( sign*beta*np.log(chi) )
        return total

    @staticmethod
    def approximate_loglikelihood(graph, average_degree, parameters: EmbeddingParameters, b=3):
        """Compute log-likelihood of differentiable S^1 model."""
        theta, kappa, beta = parameters.theta, parameters.kappa, parameters.beta
        if theta is None:
            raise ValueError("Theta cannot be None in loglikelihood.")
        if kappa is None:
            raise ValueError("Kappa cannot be None in loglikelihood.")
        if beta is None:
            raise ValueError("Beta cannot be None in loglikelihood.")

        total = 0

        n = graph.get_size()
        R_div_mu = (n * average_degree) / (beta * np.sin(np.pi / beta))
        for i in range(n - 1):
            for j in range(i + 1, n):
                chi = R_div_mu * differentiable_angular_distance(
                    theta[i], theta[j], b) / kappa[i] / kappa[j]
                sign = 1 if graph.has_edge(i, j) else -1
                total -= log1pexp( sign*beta*np.log(chi) )

        return total

    @staticmethod
    def get_mu(beta, average_degree):
        """Returns the mu value such that kappa matches the expected degree
        in the limit of inifinite size graph."""
        return beta * np.sin(np.pi / beta) / (2 * np.pi * average_degree)

    @staticmethod
    def get_H2_radii(kappa, kappa_0, mu, n=None):
        """Map kappa to hyperbolic radii in the hyperbolic polar coordinates."""
        if n is None:
            n = len(kappa)
        R = 2 * (np.log(n / (mu * np.pi)) - 2 * np.log(kappa_0))
        return R - 2 * np.log(kappa / kappa_0)


    @staticmethod
    def get_radius(n):
        """Returns the circle radius."""
        return n / (2 * np.pi)


    @staticmethod
    def align_sample(sample:EmbeddingsContainer, reference_thetas:npt.NDArray[np.double], automorphisms):
        """Align `sample` to `reference_thetas` based on the circle isometries
        and the `automorphisms` of the embedded graph. `automorphisms` is a
        list of list of indices that indicate the relabelling of vertices such
        that `new_theta = theta[automorphism]`, where `automorphism` is any
        element of `automorphisms`.
        """
        aligned = EmbeddingsContainer()
        for sample_point in sample:
            ideal_symmetry = S1Model.find_ideal_symmetry(sample_point.theta, reference_thetas, automorphisms)
            aligned.append(S1Model.apply_symmetry(sample_point, ideal_symmetry))
        return aligned


    @staticmethod
    def find_ideal_symmetry(theta, reference_theta, automorphisms) -> tuple[list[int], int, float]:
        """Returns the automorphism, rotation and flip that minimize the sum of
        squared distance between `theta` and `reference_theta`."""
        if len(automorphisms)>10:
            warnings.warn(f"There are {len(automorphisms)} automorphisms to check, optimize might"
                          " take a long time.")
        def sum_squares(shift, angles):
            distance = angular_distance(reference_theta, angle_modulo(angles+shift))
            return np.sum(distance*distance)

        best_symmetry = None  # Automorphism, orientation, shift
        smallest_error = np.inf
        if len(automorphisms) == 0:
            raise ValueError("Could not find best symmetry.")
        for permutation in automorphisms:
            permuted_theta = theta[permutation]
            for orientation in [-1, 1]:
                flipped_theta = orientation*permuted_theta
                for init in np.linspace(0, 2, 5)[:-1] * np.pi:
                    best_shift = optimize.minimize(
                            sum_squares, x0=init, bounds=((0, 2*np.pi),), args=(flipped_theta,), method="Nelder-Mead"
                        ).x
                    error = sum_squares(best_shift, flipped_theta)
                    if error < smallest_error:
                        smallest_error = error
                        best_symmetry = (permutation, orientation, best_shift)
        if best_symmetry is None:
            raise ValueError("Could not find best symmetry.")
        return best_symmetry


    @staticmethod
    def apply_symmetry(embedding:EmbeddingParameters, symmetry:tuple[list[int], int, float]):
        """Apply automorphism, rotation and flip to the embedding."""
        permutation, orientation, shift = symmetry
        if embedding.theta is not None:
            embedding.theta = angle_modulo(orientation*embedding.theta[permutation]+shift)
        if embedding.kappa is not None:
            embedding.kappa = embedding.kappa[permutation]
        return embedding


@njit
def angular_distance_1d(theta1: np.double, theta2: np.double):
    """Angular separation between `theta1` and `theta2`. Angles must be in [-pi, pi]."""
    return np.pi - np.abs(np.pi - np.abs(theta1 - theta2))

@njit
def angular_distance(theta1: npt.NDArray[np.double], theta2: npt.NDArray[np.double]):
    """Element-wise angular separation between `theta1` and `theta2`. Angles
    must be in [-pi, pi]."""
    return np.pi - np.abs(np.pi - np.abs(theta1 - theta2))

@njit
def differentiable_angular_distance(theta1, theta2, b):
    """Differentiable approximation of angular separation between `theta1` and
    `theta2`. Angles must be in [-pi, pi]."""
    return np.pi - differentiable_abs(np.pi - differentiable_abs(theta1 - theta2, b), b)

@njit
def differentiable_abs(x, b):
    """Sigmoid approximation of the absolute value function. The derivative of the function
    gets sharper around 0 as b increases. The function is the absolute value in the limit of
    infinite b."""
    return x * (2 / (1 + np.exp(-b * x)) - 1)
