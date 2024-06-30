import math 
import statistics

def manhattan_dist(r1, r2):
    """ Arguments r1 and r2 are lists of numbers """
    diff_list = []
    for (r1_elem, r2_elem) in zip(r1, r2):
        if(not(math.isnan(r1_elem) or math.isnan(r2_elem))):
            diff_list.append(float(abs(r1_elem - r2_elem)))

    if(len(diff_list) == 0):
        return math.nan
    else:
        number_missing_values = len(r1) - len(diff_list)        
        mean_distance = statistics.mean(diff_list)
        for _ in range(number_missing_values):
            diff_list.append(mean_distance)
        return sum(diff_list)

def euclidean_dist(r1, r2):
    diff_list = []
    for (r1_elem, r2_elem) in zip(r1, r2):
        if(not(math.isnan(r1_elem) or math.isnan(r2_elem))):
            diff_list.append(float((r1_elem - r2_elem)**2))

    if(len(diff_list) == 0):
        return math.nan
    else:
        number_missing_values = len(r1) - len(diff_list)        
        mean_distance = statistics.mean(diff_list)
        for _ in range(number_missing_values):
            diff_list.append(mean_distance)
        return math.sqrt(float(sum(diff_list)))

def distance_between_clusters(c1, c2, distance_fn, distance_between_clusters_measure):
    """
    Returns calculated distance between the clusters c1 and c2, using distance measure 
    distance_fn for calculating distances between elements of different clusters and using
    distance_between_clusters_measure to calculate overall distance between the clusters.
    """
    cleaned_distances = []
    for c1_elem in c1:
        for c2_elem in c2:
            distance = distance_fn(c1_elem, c2_elem)
            if(not math.isnan(distance)):
                cleaned_distances.append(distance)

    if(len(cleaned_distances) == 0):
        return math.nan
    elif(len(cleaned_distances) == 1):
        return cleaned_distances[0]
    else:
        return distance_between_clusters_measure(cleaned_distances)

def single_linkage(c1, c2, distance_fn):
    """ Arguments c1 and c2 are lists of lists of numbers
    (lists of input vectors or rows).
    Argument distance_fn is a function that can compute
    a distance between two vectors (like manhattan_dist)."""
    return distance_between_clusters(c1, c2, distance_fn, min)

def complete_linkage(c1, c2, distance_fn):
    return distance_between_clusters(c1, c2, distance_fn, max)

def average_linkage(c1, c2, distance_fn):
    return distance_between_clusters(c1, c2, distance_fn, statistics.mean)    

class HierarchicalClustering:

    def __init__(self, cluster_dist, return_distances=False):
        # the function that measures distances clusters (lists of data vectors)
        self.cluster_dist = cluster_dist

        # if the results of run() also needs to include distances;
        # if true, each joined pair in also described by a distance.
        self.return_distances = return_distances

    def unpack_cluster(self, cluster):
        """
        Returns list, with elements being the examples in the cluster.
        """
        result_list = []
        if(len(cluster) <= 1):
            return cluster
        
        for element in cluster:
            if(not isinstance(element, float)):
                result_list.extend(self.unpack_cluster(element))
        return result_list

    def closest_clusters(self, data, clusters):
        """
        Return the closest pair of clusters and their distance.
        """
        unpacked_clusters = {}
        unpacked_clusters_distances = {}
        for cluster in clusters:
            unpacked_clusters[str(cluster)] = self.unpack_cluster(cluster)
            if(not isinstance(unpacked_clusters[str(cluster)], list)):
                # if there is only one element in the cluster
                unpacked_clusters[str(cluster)] = [unpacked_clusters[str(cluster)]]
            unpacked_clusters_distances[str(cluster)] = [data[str(cluster_element)] for cluster_element in unpacked_clusters[str(cluster)]]

        distances = {}
        cleaned_distances = {}
        for i in range(0, len(clusters)):
            for j in range(i + 1, len(clusters)):
                unpacked_first_cluster = unpacked_clusters_distances[str(clusters[i])]
                unpacked_second_cluster = unpacked_clusters_distances[str(clusters[j])]
                key = f"{i} {j}"
                dist = self.cluster_dist(unpacked_first_cluster, unpacked_second_cluster)
                distances[key] = dist
                if(not math.isnan(dist)):
                    cleaned_distances[key] = dist

        if(bool(cleaned_distances)):
            closest_two_clusters = min(cleaned_distances, key=distances.get)
        else:
            closest_two_clusters = min(distances, key=distances.get)
        closest_two_clusters_list = closest_two_clusters.split()
        first_cluster = clusters[int(closest_two_clusters_list[0])]
        second_cluster = clusters[int(closest_two_clusters_list[1])]

        return first_cluster, second_cluster, distances[closest_two_clusters]
     

    def run(self, data):
        """
        Performs hierarchical clustering until there is only a single cluster left
        and return a recursive structure of clusters.
        """

        # clusters stores current clustering. It starts as a list of lists
        # of single elements, but then evolves into lists like
        # [[["Albert"], [["Branka"], ["Cene"]]], [["Nika"], ["Polona"]]]
        clusters = [[name] for name in data.keys()]

        while len(clusters) >= 2:
            first, second, distance = self.closest_clusters(data, clusters)
            # update the "clusters" variable
            new_cluster = [first, second]
            if(self.return_distances):
                new_cluster.append(distance)
            clusters.remove(first)
            clusters.remove(second)
            clusters.append(new_cluster)

        return clusters
