from dataclasses import dataclass, field
from typing import Optional

from numba import jit
import numpy as np
import numpy.typing as npt


@dataclass
class EmbeddingParameters:
    """
    Structure that contains every parameter of the S1 model. Unknown parameters are
    stored as None.
    """
    theta: Optional[npt.NDArray[np.double]] = None
    kappa: Optional[npt.NDArray[np.double]] = None
    beta: Optional[float|np.double] = None

    def __getitem__(self, key):
        """Returns parameters values of name `key`."""
        return self.__dict__[key]

    def as_dict(self):
        """Returns dictionary containing all the parameter values."""
        return {param: self[param] for param in self.names()}

    @staticmethod
    def names():
        """List of the parameter names of the S1 model."""
        return ["theta", "kappa", "beta"]

    @staticmethod
    @jit(forceobj=True)
    def concat(theta, kappa, beta):
        """Merges parameters sequentially in a 1D array."""
        if theta is None or kappa is None or beta is None:
            raise ValueError("Cannot concatenate parameters because of missing values")
        return np.concatenate((theta, kappa, [beta]))

    @staticmethod
    @jit(forceobj=True)
    def deconcat(concatenated_parameters: np.ndarray):
        """Creates `EmbeddingParameters` from 1D array."""
        n = int((np.shape(concatenated_parameters)[0]-1)/2)
        return EmbeddingParameters(theta=concatenated_parameters[:n],
                                   kappa=concatenated_parameters[n:2*n],
                                   beta=concatenated_parameters[-1])


@dataclass
class Hyperparameters:
    """Structures for the parameters of the Bayesian S1 model."""
    gamma: float
    radius: float
    beta_average: float
    beta_std: float


@dataclass
class EmbeddingsContainer:
    """Structure that contains a sequence of embeddings. Used to contain the samples."""
    thetas: list[np.ndarray] = field(default_factory=list)
    kappas: list[np.ndarray] = field(default_factory=list)
    betas: list[np.double] = field(default_factory=list)

    def append(self, parameters: EmbeddingParameters):
        """Adds new embedding to sequence."""
        if parameters.theta is not None:
            self.thetas.append(np.asarray(parameters.theta))
        if parameters.kappa is not None:
            self.kappas.append(np.asarray(parameters.kappa))
        if parameters.beta is not None:
            self.betas.append(np.double(parameters.beta))
        return self

    def extend(self, theta: list[np.ndarray]=[], kappa: list[np.ndarray]=[],
               beta: list[np.double]=[]):
        """Adds many embeddings to sequence."""
        if theta:
            self.thetas.extend(theta)
        if kappa:
            self.kappas.extend(kappa)
        if beta:
            self.betas.extend(beta)
        return self

    def __len__(self):
        """Returns longest sequence of parameters."""
        if (n := len(self.thetas)) > 0:
            return n
        if (n := len(self.kappas)) > 0:
            return n
        if (n := len(self.betas)) > 0:
            return n
        return 0

    def __iter__(self):
        """Generates the sequence of embeddings."""
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i):
        """Returns the EmbeddingParameters of the given index when the parameter exists."""
        return EmbeddingParameters(
            theta=self.thetas[i] if len(self.thetas) > 0 else None,
            kappa=self.kappas[i] if len(self.kappas) > 0 else None,
            beta=self.betas[i] if len(self.betas) > 0 else None)

    def __setitem__(self, i, values: EmbeddingParameters):
        """Changes the i-th element of the sequence of embeddings."""
        if len(self.thetas) > 0 and values.theta is not None:
            self.thetas[i] = np.asarray(values.theta)
        if len(self.kappas) > 0 and values.kappa is not None:
            self.kappas[i] = np.asarray(values.kappa)
        if len(self.betas) > 0 and values.beta is not None:
            self.betas[i] = values.beta

    def items(self):
        """Generates tuples of the parameter and sequence of its values."""
        yield "theta", self.thetas
        yield "kappa", self.kappas
        yield "beta", self.betas

    def end(self):
        """Returns the last embedding of the sequence."""
        return self[-1]


def replace_known_parameters(embedding: EmbeddingParameters,
                             known_values: Optional[EmbeddingParameters]):
    """Sets the parameters of `embedding` that are known (not `None`) in `known_values`."""
    if known_values is None:
        return embedding
    return EmbeddingParameters(
            theta=embedding.theta if known_values.theta is None else known_values.theta,
            kappa=embedding.kappa if known_values.kappa is None else known_values.kappa,
            beta=embedding.beta if known_values.beta is None else known_values.beta
        )

@dataclass
class GraphInformation:
    """Structure containing the graph information required by BIGUE."""
    n: int
    fixed_vertices: list[int]
    """First vertex angle is set to 0, second vertex angle is restricted to upper half of the circle"""
    average_degree: float

    @staticmethod
    def from_degrees(degrees):
        return GraphInformation(
            n=len(degrees),
            fixed_vertices=np.argpartition(degrees, -2)[-2:].tolist()[::-1],
            average_degree=np.average(degrees))
