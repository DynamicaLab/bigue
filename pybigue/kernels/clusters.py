from collections import Counter
import itertools
from numbers import Number

import numpy as np
from scipy.special import perm, binom
from scipy import stats

from pybigue.models import angular_distance
from pybigue.utils import angle_modulo, sample_truncated_normal
from pybigue.embedding_info import EmbeddingParameters


def flip_angle_sector(angles, theta1, theta2, crosses_pi=False):
    new_angles = angles.copy()

    theta1, theta2 = max([theta1, theta2]), min([theta1, theta2])
    if not crosses_pi:
        angles_in_sector = (angles >= theta2) & (angles <= theta1)
    else:
        angles_in_sector = (angles >= theta1) | (angles <= theta2)
        theta2, theta1 = theta1, theta2 + 2 * np.pi

    flipped_angles = angles[angles_in_sector]
    # Rotate such that theta2=0
    rotate_shift = -theta2
    flipped_angles += rotate_shift
    # Flip (theta1 is now negative) and translate such that theta1>0
    flipped_angles = -flipped_angles + theta1 + rotate_shift
    # Rotate back to original values
    flipped_angles -= rotate_shift

    new_angles[angles_in_sector] = flipped_angles
    return angle_modulo(new_angles)


def flip_horizontally(angles):
    return np.arctan2(np.sin(angles), -np.cos(angles))


def kernel_swap_clusters(fixed_vertices, gap_distribution_name):
    fixed_gap = isinstance(gap_distribution_name, Number)
    if not fixed_gap:
        gap_distribution = gap_distributions[gap_distribution_name] if gap_distribution_name is not None else NormalGapDistribution

    def kernel(current_embedding, chain_id):
        theta = np.array(current_embedding.theta)
        gap = gap_distribution_name if fixed_gap else gap_distribution.sample(len(theta))

        clusters = find_critical_gap_clusters(theta, gap)
        if clusters.number <= 2:
            return current_embedding, -np.inf

        chosen_clusters = np.random.choice(clusters.number, size=2)
        if any(fixed_vertices[0] in clusters.vertices[cluster] for cluster in chosen_clusters):
            return current_embedding, -np.inf

        proposed_theta = swap_clusters_positions(theta, clusters, *chosen_clusters)

        proposed_embedding = EmbeddingParameters(theta=proposed_theta, kappa=current_embedding.kappa, beta=current_embedding.beta)
        if fixed_gap:
            logbias = 0
        else:
            try:
                logbias = compute_cluster_bias_ratio(current_embedding, proposed_embedding, clusters, chosen_clusters, gap, gap_distribution)
            except Exception as e:
                print("swapping clusters")
                raise e
        return proposed_embedding, logbias
    return kernel


def kernel_reverse_cluster_labels(gap_distribution_name):
    fixed_gap = isinstance(gap_distribution_name, Number)
    if not fixed_gap:
        gap_distribution = gap_distributions[gap_distribution_name] if gap_distribution_name is not None else NormalGapDistribution

    def kernel(current_embedding, chain_id):
        theta = np.array(current_embedding.theta)
        gap = gap_distribution_name if fixed_gap else gap_distribution.sample(len(theta))

        clusters = find_critical_gap_clusters(theta, gap)
        chosen_cluster = np.random.randint(len(clusters.vertices))
        proposed_theta = reflect_cluster(theta, clusters.vertices[chosen_cluster])

        proposed_embedding = EmbeddingParameters(theta=proposed_theta, kappa=current_embedding.kappa, beta=current_embedding.beta)
        if fixed_gap:
            logbias = 0
        else:
            try:
                logbias = compute_cluster_bias_ratio(current_embedding, proposed_embedding, clusters, [chosen_cluster], gap, gap_distribution)
            except Exception as e:
                print("reversing clusters")
                raise e
        return proposed_embedding, logbias
    return kernel


def kernel_flip_cluster(fixed_vertices, gap_distribution_name):
    fixed_gap = isinstance(gap_distribution_name, Number)
    if not fixed_gap:
        gap_distribution = gap_distributions[gap_distribution_name] if gap_distribution_name is not None else NormalGapDistribution

    def kernel(current_embedding, chain_id):
        theta = np.array(current_embedding.theta)
        gap = gap_distribution_name if fixed_gap else gap_distribution.sample(len(theta))

        clusters = find_critical_gap_clusters(theta, gap)

        if clusters.number <= 1:
            return current_embedding, -np.inf

        chosen_cluster = np.random.choice(clusters.number)
        if fixed_vertices[0] in clusters.vertices[chosen_cluster]:
            return current_embedding, -np.inf

        proposed_theta = flip_angle_sector(theta, *clusters.boundaries[chosen_cluster],
                                           crosses_pi=clusters.crosses_pi(chosen_cluster))

        proposed_embedding = EmbeddingParameters(theta=proposed_theta, kappa=current_embedding.kappa, beta=current_embedding.beta)
        if fixed_gap:
            logbias = 0
        else:
            try:
                logbias = compute_cluster_bias_ratio(current_embedding, proposed_embedding, clusters, [chosen_cluster], gap, gap_distribution)
            except Exception as e:
                print("flipping clusters")
                raise e
        return proposed_embedding, logbias
    return kernel


def kernel_translate_cluster(gap_distribution_name):
    fixed_gap = isinstance(gap_distribution_name, Number)
    if not fixed_gap:
        gap_distribution = gap_distributions[gap_distribution_name] if gap_distribution_name is not None else NormalGapDistribution

    def kernel(current_embedding, chain_id):
        theta = np.array(current_embedding.theta)
        gap = gap_distribution if fixed_gap else gap_distribution.sample(len(theta))

        clusters = find_critical_gap_clusters(theta, gap)
        if clusters.number <= 2:
            return current_embedding, -np.inf

        chosen_clusters = np.random.choice(clusters.number, size=2)
        proposed_theta = translate_cluster(theta, clusters, *chosen_clusters)

        proposed_embedding = EmbeddingParameters(theta=proposed_theta, kappa=current_embedding.kappa, beta=current_embedding.beta)
        if fixed_gap:
            logbias = 0
        else:
            try:
                logbias = compute_cluster_bias_ratio(current_embedding, proposed_embedding, clusters, chosen_clusters, gap, gap_distribution)
            except Exception as e:
                print("translating clusters")
                raise e
        return proposed_embedding, logbias
    return kernel


def find_gap(n, prob_in_cluster):
    return np.pi*(1-(1-prob_in_cluster)**(1/(n-1)))


class UniformGapDistribution:
    inf = 0
    sup = np.pi

    @staticmethod
    def cdf(x, _):
        return x/(UniformGapDistribution.sup-UniformGapDistribution.inf)

    @staticmethod
    def sample(_):
        return (UniformGapDistribution.sup-UniformGapDistribution.inf)*np.random.random_sample() + UniformGapDistribution.inf


class NormalGapDistribution:
    prob_in_cluster = 0.9

    @staticmethod
    def cdf(x, n):
        average = find_gap(n, NormalGapDistribution.prob_in_cluster)
        params = {"loc": average, "scale": 0.5*np.pi/n}

        negative_mass = stats.norm.cdf(0, **params)
        return (stats.norm.cdf(x, **params) - negative_mass) / ( stats.norm.cdf(np.pi, **params) - negative_mass )

    @staticmethod
    def sample(n):
        average = find_gap(n, NormalGapDistribution.prob_in_cluster)
        return sample_truncated_normal(average, 0.5*np.pi/n, size=1, lower_bound=0, upper_bound=np.pi)


gap_distributions = {
        "normal": NormalGapDistribution,
        "uniform": UniformGapDistribution,
    }


def clockwise_separation(theta1, theta2):
    if isinstance(theta1, np.ndarray):
        res = theta1-theta2
        res[res<0] = (2*np.pi-(theta2-theta1))[res<0]
        return res

    if theta2 < theta1:
        return theta1-theta2
    return 2*np.pi-(theta2-theta1)


class CircularClusters:
    number: int
    vertices: list[list[int]]
    vertex_boundaries: list[tuple[float, float]]
    boundaries: list[tuple[float, float]]
    lengths: list[float]
    centers: list[float]

    def __init__(self, cluster_vertices, cluster_containing_pi, theta):
        self.vertices = cluster_vertices
        self.number = len(cluster_vertices)
        self.vertex_boundaries = []

        for i, cluster in enumerate(self.vertices):
            cluster_angles = theta[cluster]

            # If the cluster doesn't include the point ±pi,
            # left=argmax(theta) and right=argmin(theta) where theta are the
            # angles spanned by the cluster. Otherwise, left and right are
            # swapped.
            if i == cluster_containing_pi:
                # Vertex boundaries here are the most distant pair of adjacent vertices
                max_distance = 0
                position = 0
                sorted_theta = np.sort(cluster_angles)
                # j is shifted by +1 because of slice
                for j, angle in enumerate(sorted_theta[1:]):
                    distance = clockwise_separation(angle, sorted_theta[j])
                    if distance > max_distance and angle != sorted_theta[j]:
                        max_distance = distance
                        position = j

                left_boundary = sorted_theta[position]
                right_boundary = sorted_theta[position+1]
            else:
                right_boundary = np.min(cluster_angles)
                left_boundary = np.max(cluster_angles)

            self.vertex_boundaries.append((left_boundary, right_boundary))

        # "boundaries" fall in between each "vertex_boundaries", they partition
        # the circle.
        # Values must be initialized because clusters are iterated in an order
        # different from the input.
        self.boundaries = [(np.nan, np.nan) for _ in range(self.number)]
        self.centers = [np.nan for _ in range(self.number)]
        self.lengths = [np.nan for _ in range(self.number)]
        sorted_clusters = np.argsort(
            [bound[1] for bound in self.vertex_boundaries])
        for i, cluster_index in enumerate(sorted_clusters):
            # next cluster is counterclockwise
            previous_boundary = self.vertex_boundaries[sorted_clusters[
                (i - 1) % self.number]][0]
            next_boundary = self.vertex_boundaries[sorted_clusters[
                (i + 1) % self.number]][1]
            vertex_boundaries = self.vertex_boundaries[cluster_index]

            if cluster_index == cluster_containing_pi:
                # Boundaries are reversed when a cluster contains pi.
                # Reverting this order computes the angular separations in the
                # correct direction (clockwise instead of counterclockwise)
                next_boundary, previous_boundary = previous_boundary, next_boundary
                vertex_boundaries = (vertex_boundaries[1],
                                     vertex_boundaries[0])

            if previous_boundary > vertex_boundaries[
                    0]:  # the previous cluster is accessed by crossing \pi
                left_separation = abs(next_boundary - vertex_boundaries[0])
                right_separation = 2 * np.pi - abs(previous_boundary -
                                                   vertex_boundaries[1])
            elif next_boundary < vertex_boundaries[
                    1]:  # the next cluster is accessed by crossing \pi
                left_separation = 2 * np.pi - abs(next_boundary -
                                                  vertex_boundaries[0])
                right_separation = abs(previous_boundary -
                                       vertex_boundaries[1])
            else:
                left_separation = abs(next_boundary - vertex_boundaries[0])
                right_separation = abs(previous_boundary -
                                       vertex_boundaries[1])

            if cluster_index == cluster_containing_pi:
                # The next/previous and left/right are swapped because of the
                # flip done earlier.
                left_boundary = angle_modulo(previous_boundary -
                                             right_separation / 2)
                right_boundary = angle_modulo(next_boundary +
                                              left_separation / 2)
            else:
                left_boundary = angle_modulo(next_boundary -
                                             left_separation / 2)
                right_boundary = angle_modulo(previous_boundary +
                                              right_separation / 2)
            self.boundaries[cluster_index] = (left_boundary, right_boundary)

            if left_boundary < right_boundary:  # Cluster crosses ±pi
                self.centers[cluster_index] = angle_modulo(
                    (left_boundary + right_boundary) / 2 + np.pi)
                self.lengths[cluster_index] = 2 * np.pi - abs(left_boundary -
                                                              right_boundary)
            else:
                self.centers[cluster_index] = (left_boundary +
                                               right_boundary) / 2
                self.lengths[cluster_index] = abs(left_boundary -
                                                  right_boundary)

    def crosses_pi(self, cluster):
        return self.boundaries[cluster][0] < self.boundaries[cluster][1]

    def __iter__(self):
        for cluster in self.vertices:
            yield cluster

    def __getitem__(self, index):
        return self.vertices[index]


def compute_cluster_bias_ratio(current_embedding, proposed_embedding, current_clusters, clusters_involved, gap, gap_distribution):
    if isinstance(gap_distribution, Number):
        return 0

    new_clusters = None
    try:
        forward_bias = find_random_gap_logbias(current_embedding.theta, gap, current_clusters, clusters_involved, exchangeable=False, gap_distribution=gap_distribution)

        clusters_for_reverse_move = clusters_involved[::-1] # Valid because of the moves used at the moment
        new_clusters = find_critical_gap_clusters(proposed_embedding.theta, gap)
        new_clusters_indices = find_new_clusters_indices(current_clusters, new_clusters, clusters_for_reverse_move)
        backward_bias = find_random_gap_logbias(proposed_embedding.theta, gap, new_clusters, new_clusters_indices, exchangeable=False, gap_distribution=gap_distribution)
    except Exception as e:
        print("selected clusters:", clusters_involved)
        min_gap, max_gap = find_min_max_gap_generalized(np.array(current_embedding.theta), current_clusters, clusters_involved)
        print(f"[{min_gap}, {max_gap}], delta={gap}")
        print("current values")
        print("theta:", current_embedding.theta)
        print("cluster vertices:", current_clusters.vertices)
        for i in range(current_clusters.number):
            if current_clusters.crosses_pi(i):
                print(i, "crosses pi")
        print("proposed values")
        print("theta:", proposed_embedding.theta)
        if new_clusters is not None:
            print("cluster vertices:", new_clusters.vertices)
            for i in range(new_clusters.number):
                if new_clusters.crosses_pi(i):
                    print(i, "crosses pi")
        raise e
    return backward_bias-forward_bias


def find_new_clusters_indices(clusters, new_clusters, chosen_indices):
    new_chosen_indices = []
    for index in chosen_indices:
        cluster_vertices = set(clusters.vertices[index])
        for i, vertices in enumerate(new_clusters.vertices):
            if cluster_vertices == set(vertices):
                new_chosen_indices.append(i)
                break
    if len(new_chosen_indices) != len(chosen_indices):
        raise ValueError("Couldn't retrieve identical clusters.")
    return new_chosen_indices


def find_critical_gap_clusters(theta, gap):
    n = len(theta)

    clusters = []
    sorted_vertices = np.argsort(theta)

    cluster = [sorted_vertices[0]]
    cluster_end = theta[sorted_vertices[0]]
    for vertex in sorted_vertices[1:]:
        if angular_distance(theta[vertex], cluster_end) < gap:
            cluster.append(vertex)
        else:
            clusters.append(cluster)
            cluster = [vertex]
        cluster_end = theta[vertex]
    clusters.append(cluster)

    if n > 1 and len(clusters) > 1 and angular_distance(
            theta[sorted_vertices[0]], theta[sorted_vertices[-1]]) < gap:

        clusters[0].extend(clusters[-1])
        circular_clusters = CircularClusters(clusters[:-1], 0, theta)
    else:
        # Otherwise there are no counterclockwise cluster
        circular_clusters = CircularClusters(clusters, -1, theta)
    return circular_clusters


def reflect_cluster(theta, cluster_vertices):
    cluster_theta = theta[cluster_vertices]
    sorted_cluster = np.argsort(cluster_theta)[::-1]

    reflected_theta = theta.copy()
    for i, _ in enumerate(cluster_theta):
        reflected_theta[cluster_vertices[i]] = cluster_theta[sorted_cluster[i]]

    return reflected_theta


def swap_clusters_positions(theta, clusters: CircularClusters, cluster1,
                            cluster2):
    """The exchanged clusters cannot contain the fixed vertex at theta=0."""
    theta = np.copy(theta)

    closest_cluster, fartest_cluster = find_min_max_clusters(
        clusters.centers, cluster1, cluster2)
    between_clusters = find_clusters_between_clockwise(clusters.centers,
                                                       fartest_cluster,
                                                       closest_cluster)

    # Shift clusters to give room for the other cluster
    size_difference = clusters.lengths[fartest_cluster] - clusters.lengths[
        closest_cluster]
    for between_cluster in between_clusters:
        vertices = clusters.vertices[between_cluster]
        theta[vertices] = angle_modulo(theta[vertices] - size_difference)

    closest_cluster_vertices = clusters.vertices[closest_cluster]
    fartest_cluster_vertices = clusters.vertices[fartest_cluster]

    closest_shift = -find_clockwise_separation(
        clusters.boundaries[closest_cluster][1],
        clusters.boundaries[fartest_cluster][1])
    fartest_shift = find_clockwise_separation(
        clusters.boundaries[closest_cluster][0],
        clusters.boundaries[fartest_cluster][0])

    theta[closest_cluster_vertices] = angle_modulo(
        theta[closest_cluster_vertices] + closest_shift)
    theta[fartest_cluster_vertices] = angle_modulo(
        theta[fartest_cluster_vertices] + fartest_shift)

    return theta


def find_clusters_between_clockwise(centers, cluster1, cluster2):
    center1 = centers[cluster1]
    center2 = centers[cluster2]

    if center1 < center2:
        return [
            i for i, center in enumerate(centers) if center1 < center < center2
        ]
    else:
        return [
            i for i, center in enumerate(centers)
            if center > center1 or center < center2
        ]


def translate_cluster(theta, clusters: CircularClusters, cluster1, cluster2):
    theta = np.copy(theta)

    # Shifts clusters in between the insertion position
    cluster1_size = clusters.lengths[cluster1]
    distance_between_12 = 0
    for between_cluster in find_clusters_between_clockwise(
            clusters.centers, cluster1, cluster2) + [cluster2]:

        distance_between_12 += clusters.lengths[between_cluster]
        vertices = clusters.vertices[between_cluster]
        theta[vertices] = angle_modulo(theta[vertices] - cluster1_size)

    cluster1_vertices = clusters.vertices[cluster1]
    theta[cluster1_vertices] = angle_modulo(theta[cluster1_vertices] +
                                            distance_between_12)

    return theta


def find_min_max_gap(theta, clusters: CircularClusters, preserved_cluster):
    cluster_vertices = clusters.vertices[preserved_cluster]
    cluster_theta = theta[cluster_vertices]
    sorted_vertices = np.argsort(theta)

    if clusters.crosses_pi(preserved_cluster):
        counterclockwise_cluster = np.argsort(angle_modulo(cluster_theta - (np.pi+clusters.boundaries[preserved_cluster][0]) ))
    else:
        counterclockwise_cluster = np.argsort(cluster_theta)

    if len(cluster_vertices) == 1:
        delta_min = 0
    else:
        delta_min = np.max(clockwise_separation(
            cluster_theta[counterclockwise_cluster][1:],
            cluster_theta[counterclockwise_cluster][:-1]
        ))

    if len(cluster_vertices) == len(theta):
        delta_max = np.pi
    else:
        cluster_clockwise = cluster_theta[counterclockwise_cluster[0]]
        cluster_counterclockwise = cluster_theta[counterclockwise_cluster[-1]]
        outside_clockwise = theta[find_closest_outside_subseq(sorted_vertices, cluster_vertices, direction="clockwise")]
        outside_counterclockwise = theta[find_closest_outside_subseq(sorted_vertices, cluster_vertices, direction="counterclockwise")]

        clockwise_closest_dist = clockwise_separation(cluster_clockwise, outside_clockwise)
        counterclockwise_closest_dist = clockwise_separation(outside_counterclockwise, cluster_counterclockwise)

        delta_max = np.min(
            [clockwise_closest_dist, counterclockwise_closest_dist])
    return delta_min, delta_max


def find_min_max_gap_generalized(theta, clusters: CircularClusters, clusters_to_preserve):
    min_gap, max_gap = None, None

    for cluster in clusters_to_preserve:
        cluster_min_gap, cluster_max_gap = find_min_max_gap(theta, clusters, cluster)

        if min_gap is None or cluster_min_gap > min_gap:
            min_gap = cluster_min_gap
        if max_gap is None or cluster_max_gap < max_gap:
            max_gap = cluster_max_gap

    return min_gap, max_gap


def find_intermediary_gaps(theta, clusters: CircularClusters, selected_clusters, min_gap, max_gap):
    outside_cluster = list(
        itertools.chain.from_iterable([
            cluster for index, cluster in enumerate(clusters.vertices)
            if index not in selected_clusters
        ]))

    considered_theta = np.sort(theta[outside_cluster])
    return np.sort(
        list(
            filter(
                lambda delta: min_gap < delta < max_gap,
                angular_distance(considered_theta,
                                 np.roll(considered_theta, -1))))).tolist()


def find_random_gap_logbias(theta, sampled_gap, clusters: CircularClusters, selected_clusters, exchangeable, gap_distribution):
    """ This computation supposes a uniform choice of cluster.
    """
    theta = np.array(theta)
    min_gap, max_gap = find_min_max_gap_generalized(theta, clusters, selected_clusters)
    intermediary_gaps_counts = Counter(find_intermediary_gaps(theta, clusters, selected_clusters, min_gap, max_gap))
    intermediary_gaps = list(intermediary_gaps_counts.keys())
    intermediary_gaps.sort()
    n = len(theta)


    def mod_neg(x, k):
        return (x+k) % k

    proposal_logprob = 0
    I = len(intermediary_gaps_counts)+1
    cluster_sizes = np.zeros(I)

    # cluster are larger when gap is smaller
    larger_clusters = list(filter(lambda gap: gap>sampled_gap, intermediary_gaps))
    cluster_number = clusters.number
    sampled_gap_position = mod_neg(len(larger_clusters)-1, I)

    cluster_sizes[sampled_gap_position] = cluster_number
    for i, gap in enumerate(larger_clusters):
        cluster_number -= intermediary_gaps_counts[gap]
        cluster_sizes[mod_neg(sampled_gap_position+(i+1), I)] = cluster_number

    smaller_clusters = list(filter(lambda gap: gap<sampled_gap, intermediary_gaps))
    smaller_clusters.sort(reverse=True)
    cluster_number = clusters.number
    for i, gap in enumerate(smaller_clusters):
        cluster_number += intermediary_gaps_counts[gap]
        cluster_sizes[mod_neg(sampled_gap_position-(i+1), I)] = cluster_number

    for cluster_size, (lower, upper) in zip(cluster_sizes, zip([min_gap]+intermediary_gaps, intermediary_gaps+[max_gap])):
        diff = gap_distribution.cdf(upper, n) - gap_distribution.cdf(lower, n)
        if diff == 0:
            continue
        proposal_logprob += np.log(diff)
        if exchangeable:
            proposal_logprob += -np.log(binom(cluster_size, len(selected_clusters)))
        else:
            proposal_logprob += -np.log(perm(cluster_size, len(selected_clusters)))

    return proposal_logprob


def find_closest_to_zero_from_bottom(centers):
    return np.argmin(flip_horizontally(np.asarray(centers)))


def find_clockwise_separation(theta1, theta2):
    """Valid for clusters that don't contain the point theta=0. For those
    clusters, opposite sign boundaries implies that the cluster includes
    the point ±pi"""
    if theta1 * theta2 < 0:
        return 2 * np.pi - np.abs(theta1 - theta2)
    return np.abs(theta1 - theta2)


def find_min_max_clusters(centers, cluster1, cluster2):
    """Returns (min, max), where min is the cluster closest to the cluster
    containing theta=0 in the clockwise direction."""
    if 0 == find_closest_to_zero_from_bottom(
        [centers[cluster1], centers[cluster2]]):

        return cluster1, cluster2
    else:
        return cluster2, cluster1


def find_beginning_periodic(sequence, subseq):
    """Works if subseq has contiguous positions in sequence."""
    beginning = 0
    last_value_in_subseq = sequence[0] in subseq
    for i, value in reversed(list(enumerate(sequence))):
        if last_value_in_subseq and value not in subseq:
            beginning = i + 1
            break
        last_value_in_subseq = value in subseq
    return beginning


def find_end_periodic(sequence, subseq):
    """Works if subseq has contiguous positions in sequence."""
    end = len(sequence)
    last_value_in_subseq = sequence[-1] in subseq
    for i, value in enumerate(sequence):
        if last_value_in_subseq and not value in subseq:
            end = i - 1
            break
        last_value_in_subseq = value in subseq
    return end


def find_closest_outside_subseq(sorted_indices, subseq_indices, direction):
    """ Something something because obscure function. """
    n = len(sorted_indices)
    if direction == "clockwise":
        index = find_beginning_periodic(sorted_indices, subseq_indices) - 1
    elif direction == "counterclockwise":
        index = find_end_periodic(sorted_indices, subseq_indices) + 1
    else:
        raise ValueError(
            'Argument "direction" must be "counterclockwise" or "clockwise".')
    return sorted_indices[(index + n) % n]
