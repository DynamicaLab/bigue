import numpy as np
from matplotlib import pyplot as plt

from pybigue.utils import align_theta, sample_truncated_pareto, sample_uniform
from pybigue.embedding_info import EmbeddingParameters, GraphInformation
from pybigue.models import S1Model, angular_distance
from pybigue.sampling import read_sample, sample_bigue

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
# Align using highest-degree vertices
groundtruth_parameters.theta = align_theta(groundtruth_parameters.theta, *graph_info.fixed_vertices)


# Sample posterior (Note that samples are not aligned)
sample_directory = "./sample/"
# Thinning and warmup values are set to small values to have a fast (but worse) sampling
sample_bigue(adjacency_matrix, sample_directory, n_chains=4, sample_size=100, thin=30, warmup=100)
# Read sample: each chain sample is a separate key-value pair in a dictionary
samples = read_sample(sample_directory)

# Automorphisms are not computed by this package. One can use the function provided
# by pybigue-analysis which uses a custom C program built on nauty (must be compiled locally).
automorphisms = [np.arange(n)]
# Align sample to ground truth angles using model symmetries (rotations, reflection and automorphisms)
aligned_samples = {chain: S1Model.align_sample(sample, reference_thetas=groundtruth_parameters.theta, automorphisms=automorphisms)
                   for chain, sample in samples.items()}

# Compare the sampled angles to the original coordinates
fig, axes = plt.subplots(1, 2, sharey=True)
colors = ["#EEAAD7", "#9567E0", "#89B8D2", "#DFA953"]
axes[0].set_title("Without alignment")
axes[1].set_title("With alignment")
for ax, chain_samples in zip(axes, [samples, aligned_samples]):
    for i, (sample, color) in enumerate(zip(chain_samples.values(), colors)):
        thetas = np.array(sample.thetas).T
        for reference_position, ts in zip(groundtruth_parameters.theta, thetas):
            ax.scatter(np.full_like(ts, reference_position), ts, marker="o", color=color, alpha=2/len(sample))
        ax.plot([], [], label=f"Chain {i+1}", color=color, ls="none", marker="o")
    ax.set_ylabel("Sampled angle")
    ax.set_xlabel("Original angle")
plt.legend()
plt.show()
