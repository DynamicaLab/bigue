import numpy as np
import pytest

from pybigue.kernels.clusters import (CircularClusters,
                                 find_clockwise_separation,
                                 find_closest_to_zero_from_bottom,
                                 find_clusters_between_clockwise,
                                 swap_clusters_positions,
                                 translate_cluster,
                                 flip_angle_sector)
from pybigue.utils import angle_modulo


def test_angleflip_upperHalf():
    angles = np.array([0, 2, 3, 1.25, 1.75])
    flipped_angles = np.array([0, 1, 3, 1.75, 1.25])
    assert np.all(flip_angle_sector(angles, 1, 2) == flipped_angles)


def test_angleflip_lowerHalfSector():
    angles = -np.array([0, 2, 3, 1.25, 1.75])
    flipped_angles = -np.array([-0, 1, 3, 1.75, 1.25])
    assert np.all(flip_angle_sector(angles, -1, -2) == flipped_angles)


def test_angleflip_sectorCrossesOrigin():
    angles = np.array([0, -3, 2, 1.5, -1, 0, 3])
    flipped_angles = np.array([1, -3, -1, -.5, 2, 1, 3])
    assert np.all(flip_angle_sector(angles, 2, -1) == flipped_angles)


def test_angleflip_sectorCrossesPi():
    angles = np.array([0, -3, -3.1, 2, 1.5, -1, 0, 3])
    flipped_angles = np.array([0, 3, 3.1, 2, 1.5, -1, 0, -3])
    assert pytest.approx(flip_angle_sector(angles, -3, 3,
                                           crosses_pi=True)) == flipped_angles


def test_closestToZero_lowerhalf():
    assert find_closest_to_zero_from_bottom([-3, -1]) == 1


def test_closestToZero_bothhalves():
    assert find_closest_to_zero_from_bottom([-3, 1]) == 0


def test_closestToZero_upperhalf():
    assert find_closest_to_zero_from_bottom([3, 2]) == 0


def test_clustersBetweenClockwise_lowerhalf():
    assert find_clusters_between_clockwise([-1, -2, -2.1, -2.2, -3, -2.5],
                                              4, 1) == [2, 3, 5]
    assert find_clusters_between_clockwise([-1, -2, -2.1, -2.2, -3, -2.5],
                                              0, 2) == [3, 4, 5]


def test_clustersBetweenClockwise_upperhalf():
    assert find_clusters_between_clockwise([1, 2, 2.1, 2.2, 3, 2.5], 1,
                                              4) == [2, 3, 5]
    assert find_clusters_between_clockwise([1, 2, 2.1, 2.2, 3, 2.5], 3,
                                              1) == [0, 4, 5]


def test_clustersBetweenClockwise_bothhalves():
    assert find_clusters_between_clockwise([-3, 2, -1, -2, 2.5, 1], 3,
                                              1) == [2, 5]
    assert find_clusters_between_clockwise([-3, 2, -1, -2, 2.5, 1], 1,
                                              3) == [0, 4]


def test_clockwiseSeparation():
    assert find_clockwise_separation(-1, -2) == 1
    assert find_clockwise_separation(-1, 1) == pytest.approx(2 * np.pi - 2)
    assert find_clockwise_separation(3, 2) == 1


def test_swapCommunities_sideBySide():
    angles = np.array([-3, 0.5, -1, 2])
    clusters = CircularClusters([], 0, [])
    clusters.vertices = [[0, 2], [1], [3]]
    clusters.boundaries = [(-.5, 3), (.5, -.5), (3, 1)]
    clusters.centers = [2.5 / 2 - np.pi, .25, 2]
    clusters.lengths = [2 * np.pi - 3.5, 1.5, 2]
    clusters.number = len(clusters.vertices)

    assert np.all(
        np.isclose(
            swap_clusters_positions(angles, clusters, 0, 2),
            angle_modulo(np.array([-3 - 2, 0.5, -1 - 2,
                                   2 + (2 * np.pi - 3.5)]))))


def test_swapCommunities_withCommunitiesBetween():
    angles = np.array([1.5, -0.8, 0, 3, -2.5, -2])
    clusters = CircularClusters([], 0, [])
    clusters.vertices = [[0], [1], [2], [3], [4, 5]]
    clusters.boundaries = [(2.5, 0.5), (-0.5, -1), (0.5, -0.5), (-3, 2.5),
                              (-1, -3)]
    clusters.centers = [3.5 / 2, -1.5 / 2, 0, (2 * np.pi - 3 + 2.5) / 2, -2]
    clusters.lengths = [2, 0.5, 1, 2 * np.pi - 3 - 2.5, 2]
    clusters.number = len(clusters.vertices)

    assert np.all(
        np.isclose(
            swap_clusters_positions(angles, clusters, 0, 1),
            angle_modulo(
                np.array(
                    [1.5 - 3, -0.8 + 1.5, 0, 3 - 1.5, -2.5 - 1.5, -2 - 1.5]))))


def test_translateCommunity_notCrossingPi():
    angles = np.array([1.5, -0.8, 0, 3, -2.5, -2])
    clusters = CircularClusters([], 0, [])
    clusters.vertices = [[0], [1], [2], [3], [4, 5]]
    clusters.boundaries = [(2.5, 0.5), (-0.5, -1), (0.5, -0.5), (-3, 2.5),
                              (-1, -3)]
    clusters.centers = [3.5 / 2, -1.5 / 2, 0, (2 * np.pi - 3 + 2.5) / 2, -2]
    clusters.lengths = [2, 0.5, 1, 2 * np.pi - 3 - 2.5, 2]
    clusters.number = len(clusters.vertices)

    assert np.all(
        np.isclose(
            translate_cluster(angles, clusters, 1, 0),
            angle_modulo(np.array([1.5 - .5, -0.8 + 3, -.5, 3, -2.5, -2]))))


def test_translateCommunity_CrossingPi():
    angles = np.array([1.5, -0.8, 0, 3, -2.5, -2, -3.1])
    clusters = CircularClusters([], 0, [])
    clusters.vertices = [[0], [1], [2], [3, 6], [4, 5]]
    clusters.boundaries = [(2.5, 0.5), (-0.5, -1), (0.5, -0.5), (-3, 2.5),
                              (-1, -3)]
    clusters.centers = [3.5 / 2, -1.5 / 2, 0, (2 * np.pi - 3 + 2.5) / 2, -2]
    clusters.lengths = [2, 0.5, 1, 2 * np.pi - 3 - 2.5, 2]
    clusters.number = len(clusters.vertices)


    cluster3_length = 2*np.pi - (3+2.5)
    print("res", translate_cluster(angles, clusters, 3, 2))
    print("exp", angle_modulo(np.array([1.5, -0.8-cluster3_length, -cluster3_length, 3+3.5, -2.5-cluster3_length, -2-cluster3_length, -3.1+3.5])))
    assert np.all(
        np.isclose(
            translate_cluster(angles, clusters, 3, 2),
            angle_modulo(np.array([1.5, -0.8-cluster3_length, -cluster3_length, 3+3.5, -2.5-cluster3_length, -2-cluster3_length, -3.1+3.5]))))
