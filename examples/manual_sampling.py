import numpy as np

from pybigue.utils import align_theta, sample_truncated_pareto, sample_uniform, gen_cauchy_lpdf, gen_normal_lpdf
from pybigue.embedding_info import EmbeddingParameters, GraphInformation, Hyperparameters
from pybigue.models import S1Model, angular_distance
from pybigue.kernels.transforms import get_global_sampling_kernel
from pybigue.sampling import run_parallel_chains, sample_chain

np.random.seed(64)


def generate_S1_graph(parameters, average_degree):
    """Sample adjacency matrix from S^1 model."""
    beta = parameters.beta
    theta = parameters.theta
    kappa = parameters.kappa

    n = len(theta)
    R_div_mu = n * average_degree / (beta * np.sin(np.pi / beta))
    adjacency = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            chi = R_div_mu * angular_distance(
                theta[i], theta[j]) / kappa[i] / kappa[j]

            if np.random.rand() <= 1 / (1 + chi**beta):
                adjacency[i, j] = 1
                adjacency[j, i] = 1
    return adjacency

# Generating synthetic graph
n = 30
kappa_min = 4
kappa_max = 10
exponent = 2.5
beta = 2.5

groundtruth_parameters = EmbeddingParameters(
    theta=sample_uniform(-np.pi, np.pi, n),
    kappa=sample_truncated_pareto(kappa_min, kappa_max, exponent=exponent, size=n),
    beta=beta)

adjacency_matrix = generate_S1_graph(groundtruth_parameters, np.average(groundtruth_parameters.kappa))
degrees = np.sum(adjacency_matrix, axis=1)

graph_info = GraphInformation.from_degrees(degrees)
groundtruth_parameters.theta = align_theta(groundtruth_parameters.theta, *graph_info.fixed_vertices)

# Create function to compute the posterior log-pdf
hyperparameters = Hyperparameters(gamma=2.5, radius=n/(2*np.pi), beta_average=3, beta_std=2)

kappa_logprior = gen_cauchy_lpdf(0, hyperparameters.gamma)
beta_logprior = gen_normal_lpdf(hyperparameters.beta_average, hyperparameters.beta_std)

def logposterior(embedding):
    return S1Model.loglikelihood(adjacency_matrix, graph_info.average_degree, embedding.theta, embedding.kappa, embedding.beta)\
            + np.sum(kappa_logprior(embedding.kappa)) + beta_logprior(embedding.beta)


# Create sampling kernel (BIGUE kernel settings)
kernel_settings = {'random walk': {'for': ['theta', 'kappa', 'beta'], 'prob': 0.4}, 'flip': {'prob': 0.2}, 'swap': {'prob': 0.2}, 'translate': {'prob': 0.2}}
# Example of kernel that uses HMC instead of random walk:
# kernel_settings = {'hmc': {'for': ['theta', 'kappa', 'beta'], 'warmup': 200, 'prob': 0.4}, 'flip': {'prob': 0.2}, 'swap': {'prob': 0.2}, 'translate': {'prob': 0.2}}

kernel = get_global_sampling_kernel(
        kernels_settings=kernel_settings,
        init=groundtruth_parameters, # Used only for Stan warmup (not used in this case)
        adjacency=adjacency_matrix,
        graph_info=graph_info,
        logposterior=logposterior,
        hyperparameters=hyperparameters,
        known_parameters=None
    )

# Create sampling function for each chain
sample_directory = "./sample/"
sample_mcmc_chain = lambda chain_id, log_progress: sample_chain(
        kernel=kernel,
        initial_embedding_generator=lambda _: groundtruth_parameters, # Initializing with ground truth embedding
        sample_directory=sample_directory,
        sample_size=100,
        warmup=100,
        thin=10,
        chain_id=chain_id,
        log_progress=log_progress
    )
# Sample posterior (Note that samples are not aligned)
run_parallel_chains(sample_mcmc_chain, chain_number=4)
