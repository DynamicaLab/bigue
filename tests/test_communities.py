import numpy as np
import pytest

from pybigue.kernels.clusters import (
        CircularClusters, find_critical_gap_clusters, find_min_max_gap,
        find_intermediary_gaps, NormalGapDistribution, find_random_gap_logbias
    )


def iter_approx(values):
    return type(values)(map(pytest.approx, values))

def test_CircularCommunities_upperhalf():
    angles = np.array([0, 2, 3, 1.25, 1.75])
    clusters = CircularClusters([[1, 2, 4], [0, 3]], -1, angles)

    assert clusters.vertex_boundaries == [(3, 1.75), (1.25, 0)]
    boundaries = clusters.boundaries
    assert boundaries == [( (-2*np.pi+3)/2 , 1.5), (1.5, (-2*np.pi+3)/2)]
    assert clusters.centers == iter_approx([(boundaries[0][0]+boundaries[0][1])/2+np.pi, (boundaries[1][0]+boundaries[1][1])/2])
    assert clusters.lengths == iter_approx([np.pi, np.pi])

def test_CircularCommunity_lowerhalf():
    angles = np.array([-2.9, -2, -3, -1.25, -0.5])
    clusters = CircularClusters([[0, 2], [1, 3, 4]], -1, angles)

    assert clusters.vertex_boundaries == [(-2.9, -3), (-.5, -2)]

    boundaries = clusters.boundaries
    assert boundaries == [(-2.45, -.5+(2*np.pi-2.5)/2), (-.5+(2*np.pi-2.5)/2, -2.45)]
    assert clusters.centers == iter_approx([.5*(boundaries[0][0]+boundaries[0][1])+np.pi, .5*(boundaries[1][0]+boundaries[1][1])])
    assert clusters.lengths == iter_approx([2*np.pi-abs(boundaries[0][0]-boundaries[0][1]), abs(boundaries[1][0]-boundaries[1][1])])

def test_CircularCommunity_crossDiscontinuity():
    angles = np.array([3, 0, 1, -1, 1.5, -2, -0.5])
    clusters = CircularClusters([[1, 2, 4], [3, 6], [5, 0]], 2, angles)

    assert clusters.vertex_boundaries == [(1.5, 0), (-0.5, -1), (-2, 3)]
    assert clusters.boundaries == iter_approx([(2.25, -0.25), (-.25, -1.5), (-1.5, 2.25)])
    assert clusters.centers == iter_approx([1, -1.75/2, (2.25-1.5)/2-np.pi])

def test_CriticalGap():
    angles = np.array([0.24, -3, -1, -2.9, -2.5, -0.25])
    assert find_critical_gap_clusters(angles, 0.5).vertices == [[1, 3, 4], [2], [5, 0]]

def test_minmaxGap():
    angles = np.array([3, 0, 1, -1, 1.5, -2, -0.7])
    clusters = CircularClusters([[1, 2, 4], [3, 6], [5, 0]], 2, angles)

    assert iter_approx(find_min_max_gap(angles, clusters, 0)) == (1, .7)

def test_minmaxGap_crossesPi():
    angles = np.array([3, 0, 1, -1, 1.5, -2, -0.7, 3.1])
    clusters = CircularClusters([[1, 2, 4], [3, 6], [5, 0, 7]], 2, angles)

    assert iter_approx(find_min_max_gap(angles, clusters, 2)) == (2*np.pi-3.1-2, 1)

def test_minmaxGap_clusterLargerThanPi():
    cluster_theta = np.arange(-2, 3+.2, .2)
    angles = np.hstack((cluster_theta, [3.3-2*np.pi, -2.6]))

    clusters = CircularClusters([np.arange(0, len(cluster_theta)), np.arange(len(cluster_theta), len(angles))], -1, angles)

    assert iter_approx(find_min_max_gap(angles, clusters, 0)) == (0.2, 0.3)

def test_minmaxGap_entireGraph():
    angles = np.array([3, 2, 1.5, -1, -2, 3.1])
    clusters = CircularClusters([[i for i in range(len(angles))]], 0, angles)

    assert iter_approx(find_min_max_gap(angles, clusters, 0)) == (2.5, np.pi)

def test_minmaxGap_oneVertex():
    angles = np.array([-0.75, 3, -3, 0.8, -0.9, 0.6, 0.5, -2.5, -0.5, 2])
    clusters = CircularClusters([[1, 2, 7], [3, 5, 6], [4, 0, 8], [9]], 0, angles)
    assert iter_approx(find_min_max_gap(angles, clusters, 3)) == (0, 1)

def test_minmaxGap_largeClusterCrossesPi():
    """ These values were extracted from a failure in the algorithm """
    # sampled_delta = 0.36154651
    angles = np.array([ 1.57114,  -3.00664 , -2.44341,   2.42805 , -0.95689,   0.692311,  2.88641,
     -1.22667,   0.557987,  1.63353,   0.549887, -2.20525,   0.169369,  2.10322,
     -1.76899,  -0.328084,  3.11214,  -2.92056 , -1.40446,  -0.407797,  2.51804,
      0.     ,   1.46404 , -2.62289,  -0.535302,  1.03709,   0.618962,])
    clusters = CircularClusters([[1, 17, 23, 2, 11, 6, 16], [14], [18, 7, 4], [24, 19, 15, 21, 12], [10, 8, 26, 5, 25], [22, 0, 9], [13, 3, 20]], 0, angles)

    assert iter_approx(find_min_max_gap(angles, clusters, 0)) == (2.92056-2.62289, 2.88641-2.51804)

def test_intermediaryGaps():
    angles = np.array([-0.75, 3, -3, 0.8, -0.9, 0.6, 0.5, -2.5, -0.5, 2])
    clusters = CircularClusters([[1, 2, 7], [3, 5, 6], [4, 0, 8], [9]], 0, angles)

    assert iter_approx(find_min_max_gap(angles, clusters, 1)) == (0.2, 1)
    assert iter_approx(find_intermediary_gaps(angles, clusters, [1], 0.2, 1)) == [0.25, 2*np.pi-6, 0.5]


def test_intermediaryGaps_crossesPi():
    angles = np.array([-0.75, 3.1, -3, 0.8, -0.9, 0.6, 0.5, -2.5, -0.5, 2])
    clusters = CircularClusters([[1, 2], [7], [3, 5, 6], [4, 0, 8], [9]], 0, angles)

    assert iter_approx(find_min_max_gap(angles, clusters, 0)) == (2*np.pi-6.1, 0.5)
    assert iter_approx(find_intermediary_gaps(angles, clusters, [0], 2*np.pi-6.1, 0.5)) == [0.2, 0.25]


def test_logProbBias():
    angles = np.array([-0.75, 3, -3, 0.8, -0.9, 0.6, 0.5, -2.5, -0.5, 2])

    min_gap, max_gap = (0.2, 1)
    gaps = [0.25, 2*np.pi-6, 0.5]

    cdf = lambda x: NormalGapDistribution.cdf(x, len(angles))
    # log(cdf(interval)) - log(number of clusters)
    forward_prob = np.log(cdf(gaps[0]) - cdf(min_gap)) - np.log(7)\
            + np.log(cdf(gaps[1]) - cdf(gaps[0])) - np.log(6)\
            + np.log(cdf(gaps[2]) - cdf(gaps[1])) - np.log(5)\
            + np.log(cdf(max_gap) - cdf(gaps[2])) - np.log(4)

    clusters = CircularClusters([[1, 2, 7], [3, 5, 6], [4, 0, 8], [9]], 0, angles)
    assert pytest.approx(forward_prob) == \
            find_random_gap_logbias(angles, (max_gap+gaps[-1])/2, clusters, [1],
                                    exchangeable=True, gap_distribution=NormalGapDistribution)
    clusters = CircularClusters([[1], [2], [7], [3, 5, 6], [4, 0], [8], [9]], -1, angles)
    assert pytest.approx(forward_prob) == \
            find_random_gap_logbias(angles, (min_gap+gaps[0])/2, clusters, [3],
                                    exchangeable=True, gap_distribution=NormalGapDistribution)
    clusters = CircularClusters([[1], [2], [7], [3, 5, 6], [4, 0, 8], [9]], -1, angles)
    assert pytest.approx(forward_prob) == \
            find_random_gap_logbias(angles, (gaps[0]+gaps[1])/2, clusters, [3],
                                    exchangeable=True, gap_distribution=NormalGapDistribution)
