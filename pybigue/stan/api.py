from copy import deepcopy
import os
import logging

import numpy as np
from cmdstanpy import CmdStanModel

from pybigue.models import S1Model
from pybigue.embedding_info import EmbeddingsContainer, GraphInformation, EmbeddingParameters, replace_known_parameters


file_dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

# Hide cmdstanpy stdout info
cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = False
# remove all existing handlers
cmdstanpy_logger.handlers = []
cmdstanpy_logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('stan.log')
cmdstanpy_logger.addHandler(handler)


def get_stan_kernel(inferred_parameters: list[str], known_parameters, adjacency, graph_info:GraphInformation, hyperparameters, initial_warmup, b):
    if b is None:
        b = 3
    metric = None
    step_size = None
    graph_data = get_graph_stan_data(adjacency, graph_info)

    stan_model = CmdStanModel(stan_file=get_s1_stan_model_path(inferred_parameters))
    def hmc_sample(current_embedding, chain_id, sample_size=None, thin=None):
        nonlocal metric, step_size

        if metric is None:
            stan_initial_values = None
        else:
            stan_initial_values = get_s1_stan_init(
                initial_parameters=current_embedding,
                fixed_vertices=graph_info.fixed_vertices,
                inferred_parameters=inferred_parameters
                )

        embedding_copy = deepcopy(replace_known_parameters(current_embedding, known_parameters))
        complete_data = graph_data | get_s1_embedding_data(embedding_copy, inferred_parameters, hyperparameters, graph_info, b)

        sample_args = {
            "data": complete_data,
            "inits": stan_initial_values,
            "show_progress": sample_size is not None,
            "iter_sampling": 1 if sample_size is None else sample_size,
            "thin": thin,
            "chains": 1,
            "time_fmt": "%Y%m%d%H%M%S_" + str(chain_id),
        }
        if metric is not None and step_size is not None:
            sample_args.update({
                "iter_warmup": 0,
                "metric": metric,
                "step_size": step_size ,
                "adapt_engaged": False
            })
        else:
            print("Stan warmup")
            sample_args["iter_warmup"] = initial_warmup

        stan_output = stan_model.sample(**sample_args)
        if metric is None or step_size is None:
            metric = [{
                "inv_metric": metric
            } for metric in stan_output.metric]
            step_size = list(map(float, stan_output.step_size))

        embedding = unpack_parameters(stan_output, graph_info)
        if sample_size is None:
            return EmbeddingParameters(**{
                param: current_embedding[param] if values == [] else values[0]
                for param, values in embedding.items()}), 0
        return embedding, 0
    return hmc_sample


def unpack_parameters(stan_output, graph_information):
    parameters_dataframe = stan_output.draws_pd()
    get_param_array = lambda param_format: get_parameters_from_dataframe(
        parameters_dataframe, param_format, graph_information.n)

    theta_inferred = "theta[1]" in parameters_dataframe.columns
    kappa_inferred = "kappa[1]" in parameters_dataframe.columns
    beta_inferred = "beta_" in parameters_dataframe.columns
    return EmbeddingsContainer(
        thetas=get_param_array("theta[{}]") if theta_inferred else [],
        kappas=get_param_array("kappa[{}]") if kappa_inferred else [],
        betas=parameters_dataframe["beta_"].tolist() if beta_inferred else []
    )


def get_graph_stan_data(adjacency, graph_info: GraphInformation):
    data = {}
    n = graph_info.n

    data["N"] = n
    data["average_degree"] = graph_info.average_degree
    data["edge"] = [
        int(adjacency[i, j]) for i in range(n) for j in range(i + 1, n)
    ]
    # Stan indices start at 1
    data["fixed_vertices"] = np.array(graph_info.fixed_vertices) + 1

    return data


def get_s1_embedding_data(current_embedding: EmbeddingParameters, inferred_parameters, hyperparameters, graph_info, b):
    stan_data = {}
    if "theta" in inferred_parameters:
        stan_data["b"] = b
    else:
        stan_data["theta"] = current_embedding.theta

    if "kappa" in inferred_parameters:
        stan_data["gamma_"] = hyperparameters.gamma
    else:
        stan_data["kappa"] = current_embedding.kappa

    if "beta" in inferred_parameters:
        stan_data["average_degree"] = graph_info.average_degree
        stan_data["beta_average"] = hyperparameters.beta_average
        stan_data["beta_std"] = hyperparameters.beta_std
    else:
        mu = S1Model.get_mu(current_embedding.beta, graph_info.average_degree)
        stan_data["beta_"] = current_embedding.beta
        stan_data["radius_div_mu"] = graph_info.n / (2 * np.pi * mu)

    return stan_data


def get_s1_stan_init(initial_parameters: EmbeddingParameters, fixed_vertices, inferred_parameters):
    if initial_parameters is None:
        return {}
    init = {}
    if "theta" in inferred_parameters:
        theta_1 = initial_parameters.theta[fixed_vertices[1]]
        theta = [
            a for i, a in enumerate(initial_parameters.theta)
            if i not in fixed_vertices
        ]
        init["restricted_vertex"] = [np.cos(theta_1), np.sin(theta_1)]
        init["vertex_positions"] = np.array(
            [np.cos(np.array(theta)),
             np.sin(np.array(theta))]).T

    if "kappa" in inferred_parameters:
        init["kappa"] = initial_parameters.kappa
    if "beta" in inferred_parameters:
        init["beta_"] = initial_parameters.beta
    return init


def get_s1_stan_model_path(inferred_parameters):
    model_name = "s1"
    if "theta" in inferred_parameters:
        model_name += "_theta"
    if "kappa" in inferred_parameters:
        model_name += "_kappa"
    if "beta" in inferred_parameters:
        model_name += "_beta"

    return os.path.join(file_dir_path, model_name+".stan")


def get_parameters_from_dataframe(sample, parameter, n):
    if parameter.format(1) not in sample.columns:
        raise ValueError(
            f"Parameter {parameter.format(1)} not found in sample.")

    return (np.array([sample[parameter.format(i + 1)]
                      for i in range(n)]).T).tolist()
