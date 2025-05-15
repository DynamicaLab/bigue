import numpy as np

from pybigue.embedding_info import EmbeddingsContainer, EmbeddingParameters


def test_append_parametersAdded():
    n=10
    container = EmbeddingsContainer([np.zeros(n)], [np.ones(n)])
    new_theta = np.full(n, 0.5)
    new_kappa = np.full(n, 2)
    container.append(EmbeddingParameters(theta=new_theta, kappa=new_kappa))

    assert np.all( np.array(container.thetas)==np.array([np.zeros(n), new_theta]) )
    assert np.all( np.array(container.kappas)==np.array([np.ones(n), new_kappa]) )

def test_extend_addsParameters():
    n=10
    container = EmbeddingsContainer([np.zeros(n)], [np.ones(n)])
    new_theta = [np.full(n, i) for i in range(1, 5)]
    new_kappa = [np.full(n, -i) for i in range(1, 5)]
    container.extend(new_theta, new_kappa)

    assert np.all( np.array(container.thetas)==np.array([np.zeros(n)]+new_theta) )
    assert np.all( np.array(container.kappas)==np.array([np.ones(n)]+new_kappa) )

def test_end_returnsLastParameters():
    n=10
    last_theta = np.full(n, 0.5)
    last_kappa = np.full(n, 5)
    container = EmbeddingsContainer([np.zeros(n), last_theta], [np.ones(n), last_kappa])
    last = container.end()
    assert np.all(last.theta == last_theta)
    assert np.all(last.kappa == last_kappa)
