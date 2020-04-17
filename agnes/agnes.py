import pandas as pd
import numpy as np
import math, time

class Agnes :
    def __init__(self):
        self.n_clusters = 0
        self.dataframe = None
        self.clusters = []
        self.distance_matrix = None
        self.linkage = -1
        self.clusters_append = []
        self.Z = []
        self.final_clusters = []
    
    def euclidean_distance(self, X, Y) :
        ''' Calculate the Euclidean distance of two vectors '''
        return math.sqrt(sum([(a - b)**2 for a, b in zip(X, Y)]))

    def get_dist_cluster(self, id_cluster1, id_cluster2):
        ''' Get best data in cluster for calculate distance '''
        if self.linkage == 0: # single linkage
            min_dist = math.inf
            for node1 in self.clusters[id_cluster1]:
                data1 = self.dataframe.iloc[node1]
                for node2 in self.clusters[id_cluster2]:
                    # Cari data-data yang paling deket
                    data2 = self.dataframe.iloc[node2]
                    temp = self.euclidean_distance(data1, data2)
                    # Update biar selalu paling kecil
                    if temp < min_dist: min_dist = temp
            return min_dist
        
        elif self.linkage == 1: # complete linkage
            max_dist = -1
            for node1 in self.clusters[id_cluster1]:
                data1 = self.dataframe.iloc[node1]
                for node2 in self.clusters[id_cluster2]:
                    # Cari data-data yang paling jauh
                    data2 = self.dataframe.iloc[node2]
                    temp = self.euclidean_distance(data1, data2)
                    # Update biar selalu paling besar
                    if temp > max_dist: max_dist = temp
            return max_dist
        
        elif self.linkage == 2: # average linkage
            sum_dist = 0
            for node1 in self.clusters[id_cluster1]:
                data1 = self.dataframe.iloc[node1]
                for node2 in self.clusters[id_cluster2]:
                    data2 = self.dataframe.iloc[node2]
                    # Jumlahkan jarak 
                    sum_dist += self.euclidean_distance(data1, data2)
            return sum_dist / (len(self.clusters[id_cluster1]) * len(self.clusters[id_cluster2]))
        
        elif self.linkage == 3: # average group linkage
            sum1, sum2 = 0, 0
            for node1 in self.clusters[id_cluster1]:
                data1 = self.dataframe.iloc[node1]
                sum1 += data1
            for node2 in self.clusters[id_cluster2]:
                data2 = self.dataframe.iloc[node2]
                sum2 += data2    
            mean1 = sum1 / len(self.clusters[id_cluster1])
            mean2 = sum2 / len(self.clusters[id_cluster2])
            return self.euclidean_distance(mean1, mean2)

    def calc_distance_mat(self):
        n_data = self.n_clusters
        # init with zero
        dist_matrix = np.zeros((n_data, n_data))

        for i in range(n_data):
            for j in range(n_data):
                if dist_matrix[j][i] == 0:
                    if i != j:
                        dist_matrix[i][j] = self.get_dist_cluster(i, j)
                    else:
                        dist_matrix[i][j] = math.inf
                else:
                    dist_matrix[i][j] = dist_matrix[j][i]
        return dist_matrix

    def set_linkage(self, linkage_type):
        if linkage_type.lower() == 'single':
            return 0
        elif linkage_type.lower() == 'complete':
            return 1
        elif linkage_type.lower() == 'average':
            return 2
        elif linkage_type.lower() == 'average group':
            return 3
        return -1

    def get_min_dist(self):
        min_val = math.inf
        node1, node2 = -1, -1
        for i in range(len(self.distance_matrix)):
            temp = min(self.distance_matrix[i])
            if temp < min_val:
                min_val = temp
                node1, node2 = i, np.argmin(self.distance_matrix[i])
        return min_val, node1, node2
        
    def join_cluster(self, node1, node2, dist):
        clusters = self.clusters
        cluster1 = clusters.pop(min(node1, node2))
        cluster2 = clusters.pop(max(node1, node2) -1)
        id_c1 = self.clusters_append.index(cluster1)
        id_c2 = self.clusters_append.index(cluster2)
        n_singleton = len(cluster1) + len(cluster2)
        self.Z.append([float(id_c1), float(id_c2), float(dist), float(n_singleton)])
        self.clusters_append += [cluster1 + cluster2]
        return clusters + [cluster1 + cluster2]

    def agnes(self, dataframe, linkage_type, k=1):
        self.dataframe = dataframe
        self.linkage = self.set_linkage(linkage_type)
        
        if self.linkage == -1:
            print("Use 'single', 'complete', 'average', or 'average group'")
            return

        # Init n-clusters
        self.n_clusters = self.dataframe.shape[0]
        self.clusters = [[i] for i in range(self.n_clusters)]
        self.clusters_append = [i for i in self.clusters]

        # print(self.clusters)
        # print(self.clusters_append)
        
        iter = 1
        start_tot = time.time()
        while self.n_clusters > 1:
            start = time.time()
            self.distance_matrix = self.calc_distance_mat()
            # print("matriks jarak \n", self.distance_matrix)
            min_val, node1, node2 = self.get_min_dist()
            # print("minimal value", min_val, "from", node1, node2)
            self.clusters = self.join_cluster(node1, node2, min_val)
            print("cluster baru iter", iter, self.clusters_append[-1])
            self.n_clusters = len(self.clusters)
            end = time.time()
            print("waktu iterasi", iter, end-start, "s")
            # print("cluster append >", self.clusters_append)
            # print("Z >\n", np.array(self.Z))
            if self.n_clusters == k:
                print("\niterasi dilanjutkan untuk dendrogram")
                self.final_clusters = [c for c in self.clusters]
            iter +=1
        
        end_tot = time.time()
        print("Total Time", end_tot-start_tot, "s")
        return np.array(self.Z)

    def get_label(self):
        n_data = self.dataframe.shape[0]
        label = []
        for i in range(n_data):
            for id_cluster in range(len(self.final_clusters)):
                if i in self.final_clusters[id_cluster]:
                    label.append(id_cluster)
        return label
