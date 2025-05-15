import os
import re
import multiprocessing

import numpy as np

from pybigue.embedding_info import EmbeddingsContainer, EmbeddingParameters, GraphInformation, Hyperparameters
from pybigue.utils import align_theta, gen_cauchy_lpdf, gen_normal_lpdf, sample_uniform, sample_truncated_normal
from pybigue.kernels.transforms import get_global_sampling_kernel
from pybigue.models import S1Model


def sample_bigue(adjacency_matrix, sample_directory, sample_size=500, n_chains=4, warmup=10, thin=10000, initial_parameters_generator=None, verbose=True) -> None:
    """Default BIGUE sampler. Writes the sampled parameters in numpy binaries in `sample_directory`.

    The default values for `warmup` and `thinning` are those used in the paper, but they should ideally increased for graphs larger than ~50 vertices.

    Arguments:
        adjacency_matrix: graph's adjacency matrix.
        sample_directory: output path for the samples (the directory must exist).
        sample_size: sample size of each chain.
        n_chains: number of parallel chains to use.
        warmup: number of iterations to discard at the beginning of the chain .
        thin: number of iterations discarded between sample points (minus 1) to reduce autocorrelation.
        initial_parameters_generator: function that generates the initialization given the chain id. Default: kappa set to degrees, theta and beta sampled from prior.
        verbose: display progress if `True`.
    """
    degrees = np.sum(adjacency_matrix, axis=1)
    n = len(degrees)
    graph_info = GraphInformation.from_degrees(degrees)
    hyperparameters = Hyperparameters(gamma=2.5, radius=n/(2*np.pi), beta_average=3, beta_std=2)

    kappa_logprior = gen_cauchy_lpdf(0, hyperparameters.gamma)
    beta_logprior = gen_normal_lpdf(hyperparameters.beta_average, hyperparameters.beta_std)
    # theta prior constant

    def logposterior(embedding):
        return S1Model.loglikelihood(adjacency_matrix, np.average(degrees), embedding.theta, embedding.kappa, embedding.beta)\
                + np.sum(kappa_logprior(embedding.kappa)) + beta_logprior(embedding.beta)

    if initial_parameters_generator is None:
        def init(_):
            # The sampling algorithm seeds the generator so different chains have different initializations
            embedding = EmbeddingParameters(
                            theta=sample_uniform(-np.pi, np.pi, n),
                            kappa=np.array([max(1e-10, deg) for deg in degrees]),
                            beta=sample_truncated_normal(hyperparameters.beta_average, hyperparameters.beta_std, size=1, lower_bound=1))
            embedding.theta = align_theta(embedding.theta, *graph_info.fixed_vertices)
            return embedding
    else:
        init = initial_parameters_generator


    kernel_settings = {'random walk': {'for': ['theta', 'kappa', 'beta'], 'prob': 0.4}, 'flip': {'prob': 0.2}, 'swap': {'prob': 0.2}, 'translate': {'prob': 0.2}}
    kernel = get_global_sampling_kernel(
            kernels_settings=kernel_settings,
            init=EmbeddingParameters(),  # Necessary only for Stan warmup
            adjacency=adjacency_matrix,
            graph_info=graph_info,
            logposterior=logposterior,
            hyperparameters=hyperparameters,
            known_parameters=None
        )

    sample_mcmc_chain = lambda chain_id, log_progress: sample_chain(
            kernel=kernel,
            initial_embedding_generator=init,
            sample_directory=sample_directory,
            sample_size=sample_size,
            warmup=warmup,
            thin=thin,
            chain_id=chain_id,
            log_progress=log_progress if verbose else lambda *_: None
        )
    run_parallel_chains(sample_mcmc_chain, chain_number=n_chains)


def get_progress_objects(n_jobs):
    manager = multiprocessing.Manager()
    ns = manager.Namespace()
    ns.thread_states = [0 for _ in range(n_jobs)]
    lock = multiprocessing.Lock()
    return manager, ns, lock


def update_progress(ns, lock, progress_id, n, state):
    with lock:
        tmp = ns.thread_states
        tmp[progress_id] = state
        ns.thread_states = tmp
        display_progress(ns.thread_states, n)


def display_progress(thread_states, n):
    print("".join([f"[{i}]:{str(int(state/n*100))+'%':<6}"
                   for i, state in enumerate(thread_states)]),
          end="\r", flush=True)

chain_file_regex = re.compile(r".+_(\d)+\.npy")
sample_file_format = "{}_{}.npy"


def run_parallel_chains(sample_func, chain_number):
    _, ns, lock = get_progress_objects(chain_number)

    jobs = []
    for chain_id in range(chain_number):
        log_progress = lambda chain_id, iteration, sample_size: \
                update_progress(ns, lock, chain_id, sample_size, iteration)

        job = multiprocessing.Process(target=sample_func,
                                      args=(chain_id, log_progress))
        job.start()
        jobs.append(job)

    for job in jobs:
        job.join()


def sample_chain(
    kernel,
    initial_embedding_generator,
    sample_directory,
    sample_size,
    warmup,
    thin,
    chain_id=0,
    merge_samples=False,
    log_progress=lambda chain_id, iteration, sample_size: print(
        f"chain_id {chain_id+1}: {iteration/sample_size*100:.2f}%", end="\r")):
    """
    params:
        special_proposals: list of pairs (function, probability) of moves.
    """
    if sample_directory is not None and not os.path.isdir(sample_directory):
        raise OSError(f"The sampling directory \"{sample_directory}\" does not exist.")

    # Advance rng to have different chains
    for _ in range(chain_id):
        np.random.rand()

    embedding = initial_embedding_generator(chain_id)
    sample = EmbeddingsContainer()

    actual_sample_size = thin*(sample_size+warmup)
    log_progress(chain_id, warmup, sample_size)

    for iteration in range(actual_sample_size):
        embedding = kernel(embedding, chain_id)
        if iteration >= warmup*thin and ((iteration-warmup*thin) % thin) == 0:
            sample.append(embedding)

        if iteration % thin == 0 or iteration == actual_sample_size:
            log_progress(chain_id, iteration/thin-warmup+1, sample_size)

    if sample_directory is not None:
        write_sample(sample, chain_id, sample_directory, merge=merge_samples)
    return sample


def sample_hmc(
        kernel,
        initial_embedding_generator,
        sample_directory,
        sample_size,
        thin,
        chain_id,
        merge_samples=False):
    embedding = initial_embedding_generator(chain_id)
    # Advance rng to have different chains
    for _ in range(chain_id):
        np.random.rand()

    if thin is None or thin <= 0:
        thin = 1
    sample = kernel(embedding, chain_id, sample_size=sample_size*thin, thin=thin)
    if sample_directory is not None:
        write_sample(sample, chain_id, sample_directory, merge=merge_samples)
    return sample


def write_sample(sample: EmbeddingsContainer, chain_id: int, directory: str, file_format=sample_file_format, merge=False):
    for parameter, values in sample.items():
        file_name = file_format.format(parameter, chain_id)
        file_path = os.path.join(directory, file_name)

        previous_sample = []
        if os.path.isfile(file_path):
            if merge:
                previous_sample = [np.load(file_path)]
            else:
                os.remove(file_path)
        if len(values) > 0:
            np.save(file_path, np.concatenate(previous_sample+[np.stack(values)]))


def read_sample(sample_directory, file_format=sample_file_format):
    chain_samples = {}
    if not os.path.isdir(sample_directory):
        return chain_samples

    chain_regex = file_format.format("(:?theta|kappa|beta)", r"(\d)+")
    chains = set(filter(lambda x: x is not None, map(lambda x: re.match(chain_regex, x), os.listdir(sample_directory))))
    for match in chains:
        chain_id = match[2]
        sample = {}
        for parameter, _ in EmbeddingsContainer().items():
            parameter_filename = os.path.join(
                sample_directory,
                file_format.format(parameter, chain_id))
            sample[parameter+"s"] = np.load(parameter_filename) if os.path.isfile(parameter_filename) else None
        chain_samples[chain_id] = EmbeddingsContainer(**sample)
    return chain_samples
