from typing import Optional
import numpy as np

from pybigue.utils import align_theta
from pybigue.stan.api import get_stan_kernel
from pybigue.embedding_info import GraphInformation, EmbeddingParameters, Hyperparameters
from .random_walk import kernel_theta_rw, kernel_1d_theta_rw, kernel_kappa_rw, kernel_beta_rw
from .clusters import kernel_flip_cluster, kernel_swap_clusters, kernel_translate_cluster, kernel_reverse_cluster_labels


def combine_kernels(subkernels, subkernel_probs=None):
    def combined_kernel(*args, **kwargs):
        kernel = np.random.choice(subkernels, p=subkernel_probs)
        return kernel(*args, **kwargs)
    return combined_kernel


def mh_adjustment(logposterior, fixed_vertices):
    def decorater(proposal_distribution):
        def accept_proposal(current_embedding, chain_id):
            proposed_embedding, logbias = proposal_distribution(
                current_embedding=current_embedding, chain_id=chain_id)
            if not np.isfinite(logbias):
                return current_embedding
            if proposed_embedding.theta is not None:
                proposed_embedding.theta = align_theta(proposed_embedding.theta, *fixed_vertices)

            current_logposterior = logposterior(current_embedding)
            proposed_logposterior = logposterior(proposed_embedding)
            acceptance_probability = np.exp(proposed_logposterior -
                                            current_logposterior + logbias)
            if np.isnan(acceptance_probability) or np.random.rand() > acceptance_probability:
                return current_embedding
            return proposed_embedding
        return accept_proposal
    return decorater


def get_global_sampling_kernel(kernels_settings: dict,
                               init:EmbeddingParameters,
                               known_parameters:Optional[EmbeddingParameters],
                               adjacency: np.ndarray,
                               graph_info:GraphInformation,
                               hyperparameters:Hyperparameters,
                               logposterior):
    transform = lambda kernel: mh_adjustment(logposterior, graph_info.fixed_vertices)(kernel)
    return get_global_transformed_kernel(kernels_settings, init, known_parameters,
                                         adjacency, graph_info, hyperparameters, transform,
                                         transform_hmc=False)


def get_global_transformed_kernel(kernels_settings: dict,
                                  init:EmbeddingParameters,
                                  known_parameters:Optional[EmbeddingParameters],
                                  adjacency: np.ndarray,
                                  graph_info:GraphInformation,
                                  hyperparameters:Hyperparameters,
                                  transform,
                                  transform_hmc=True,
                                  ):
    kernels = []
    probs = []
    for kernel_name, settings in kernels_settings.items():
        if kernel_name == "hmc":
            # '_' required: new_kernel is evaluated when called and kernel gets overwritten in other iterations.
            _kernel = get_stan_kernel(settings["for"], known_parameters, adjacency, graph_info, hyperparameters,
                                     settings.get("warmup"), settings.get("b"))

            _kernel(init, 0) # warmup hmc
            new_kernel = transform(_kernel) if transform_hmc else lambda *args, **kwargs: _kernel(*args, **kwargs)[0]
            kernels.append(new_kernel)

        else:
            match kernel_name:
                case "random walk":
                    kernel = get_random_walk_kernel(settings["for"], graph_info,
                                                    settings.get("std"), settings.get("subkernel probs"))

                case "random walk 1d":
                    kernel = get_random_walk_kernel(settings["for"], graph_info,
                                                    settings.get("std"), settings.get("subkernel probs"), one_dim=True)

                case "swap":
                    kernel = kernel_swap_clusters(graph_info.fixed_vertices, settings.get("gap_distribution"))

                case "translate":
                    kernel = kernel_translate_cluster(settings.get("gap_distribution"))

                case "reverse":
                    kernel = kernel_reverse_cluster_labels(settings.get("gap_distribution"))

                case "flip":
                    kernel = kernel_flip_cluster(graph_info.fixed_vertices, settings.get("gap_distribution"))

                case _:
                    raise NotImplementedError(f"Kernel {kernel_name} does not exist.")

            kernels.append(transform(kernel))
        probs.append(settings["prob"])

    return combine_kernels(kernels, probs)


def get_random_walk_kernel(inferred_parameters: list[str], graph_info: GraphInformation,
                           std:float, subkernel_probs=None, one_dim=False):
    subkernels = []
    for parameter in inferred_parameters:
        match parameter:
            case "theta":
                if one_dim:
                    subkernels.append(kernel_1d_theta_rw(graph_info.n, std))
                else:
                    subkernels.append(kernel_theta_rw(graph_info.n, std))
            case "kappa":
                subkernels.append(kernel_kappa_rw(graph_info.n, std))
            case "beta":
                subkernels.append(kernel_beta_rw(graph_info.n))
            case _:
                raise ValueError(f"No random walk kernel defined for \"{parameter}\".")
    return combine_kernels(subkernels, subkernel_probs)
