# BIGUE hyperbolic embedder

<img src="https://raw.githubusercontent.com/DynamicaLab/bigue/refs/heads/main/logo.png" style="float: right;" alt="isolated" width="200"/>

Bayesian Inference of a Graph's Unkown Embedding (BIGUE) is a Markov Chain Monte Carlo algorithm that samples the best hyperbolic embeddings of a given network according to a Bayesian $\mathbb{S}^1$ model.

The code used to generate the Figures in the paper is in the [bigue-analysis] repository.
If you use this algorithm, please cite:

[Simon Lizotte](https://siliz4.github.io), [Jean-Gabriel Young](https://jgyoung.ca) and [Antoine Allard](https://antoineallard.github.io)
"Symmetry-driven embedding of networks in hyperbolic space". _Commun. Phys._ __8__, 199 (2025).
https://doi.org/10.1038/s42005-025-02122-0

## Installation

### Dependencies

The pybigue library requires:

- Python (>= 3.10)
- NumPy
- SciPy
- CmdStanPy
- Numba


pybigue can be installed directly from [PyPi](https://pypi.org/)
```
pip install pybigue
```

## Usage

To use the default BIGUE sampler,
```{python}
from pybigue.sampling import sample_bigue, read_sample

# ... code that obtains an numpy adjacency matrix

sample_directory = "./sample/"
sample_bigue(adjacency_matrix, sample_directory) # Writes samples to disk
samples = read_sample(sample_directory) # Reads samples
```

Two complete examples are presented in the `examples`. The file `embed_synthetic_graph.py` uses the default BIGUE algorithm (requires also `matplotlib`) and the file `manual_sampling.py` shows how one can tweak the sampler (hyperparameter values, prior distributions, subkernel selection probabilities, combining with Stan HMC, etc.).

Note that the __samples are not aligned automatically__. This Python library doesn't provide a tool to compute automorphisms. This can be found in the [bigue-analysis] repository.
Nevertheless, the samples can be aligned given a list of automorphisms

```{python}
from pybigue.models import S1Model

# Each automorphism is an array of the new label of each vertex.
automorphisms = [np.arange(adjacency_matrix.shape[0])] # Trivial automorphism
aligned_samples = {
    chain: S1Model.align_sample(sample, reference_thetas=some_thetas, automorphisms=automorphisms)
        for chain, sample in samples.items()
}
```


## Source code

The lastest version can be fetched with
```
git clone https://github.com/DynamicaLab/bigue.git
```

## Testing

After downloding the code, the test suite can be ran using (requires installing `pytest`)
```{python}
pytest pybigue
```

[bigue-analysis]: https://github.com/DynamicaLab/bigue-analysis
