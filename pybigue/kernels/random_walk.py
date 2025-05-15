import numpy as np

from pybigue.utils import angle_modulo, sample_truncated_normal, truncated_normal_lpdf
from pybigue.embedding_info import EmbeddingParameters


def kernel_theta_rw(vertex_number, std=None):
    if std is None:
        std = 0.5*np.pi/vertex_number

    def kernel(current_embedding, chain_id):
        return EmbeddingParameters(theta=angle_modulo(
            sample_truncated_normal(current_embedding.theta, std, vertex_number,
                                    lower_bound=-np.pi, upper_bound=np.pi)
            ),
                                   kappa=current_embedding.kappa,
                                   beta=current_embedding.beta), 0
    return kernel

def kernel_1d_theta_rw(vertex_number, std=None):
    if std is None:
        std = 0.5*np.pi/vertex_number

    def kernel(current_embedding, chain_id):
        proposed_theta = np.copy(current_embedding.theta)
        moved_vertex = np.random.randint(0, vertex_number)
        proposed_theta[moved_vertex] = angle_modulo(
                proposed_theta[moved_vertex] +
                sample_truncated_normal(0, std, size=1, lower_bound=-np.pi, upper_bound=np.pi)
            )
        return EmbeddingParameters(theta=proposed_theta,
                                   kappa=current_embedding.kappa,
                                   beta=current_embedding.beta), 0
    return kernel


def kernel_kappa_rw(vertex_number, std):
    if std is None:
        std = 0.5

    def kernel(current_embedding, chain_id):
        proposed_kappa = sample_truncated_normal(current_embedding.kappa, std, size=vertex_number, lower_bound=1e-10)
        forward_logbias = np.sum(truncated_normal_lpdf(proposed_kappa, current_embedding.kappa, std, lower_bound=1e-10))
        backwards_logbias = np.sum(truncated_normal_lpdf(current_embedding.kappa, proposed_kappa, std, lower_bound=1e-10))
        return EmbeddingParameters(theta=current_embedding.theta,
                                   kappa=proposed_kappa,
                                   beta=current_embedding.beta), backwards_logbias-forward_logbias
    return kernel


def kernel_beta_rw(std):
    if std is None:
        std = 0.3

    def kernel(current_embedding, chain_id):
        proposed_beta = sample_truncated_normal(current_embedding.beta, std, size=1, lower_bound=1)
        forward_logbias = truncated_normal_lpdf(proposed_beta, current_embedding.beta, std, lower_bound=1)
        backwards_logbias = truncated_normal_lpdf(current_embedding.beta, proposed_beta, std, lower_bound=1)
        return EmbeddingParameters(theta=current_embedding.theta,
                                   kappa=current_embedding.kappa,
                                   beta=proposed_beta), np.sum(backwards_logbias-forward_logbias)
    return kernel
