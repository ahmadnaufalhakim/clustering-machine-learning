import pandas as py
import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
import copy
import math
warnings.filterwarnings("ignore")

random.seed(13517055)

class K_Means :
    def __init__(self):
        self.n_clusters = 0
        self.centroids = []
        self.clusters = {}

    def get_n_clusters(self) :
        """Get number of clusters"""
        return self.n_clusters

    def set_n_clusters(self, n_clusters) :
        """Set number of clusters"""
        self.n_clusters = n_clusters

    def euclidean_distance(self, X, Y) :
        """Calculate the Euclidean distance of two vectors"""
        return math.sqrt(sum([(a - b)**2 for a, b in zip(X, Y)]))

    def encode_categorical_feature(self, dataframe) :
        """Numerize categorical column data"""
        df = dataframe
        for column in df.columns :
            if df[column].dtypes != "int64" and df[column].dtypes != "float64" :
                feature_values = {}
                for index, value in enumerate(df[column].unique()) :
                    feature_values[value] = index
                df = df.replace(feature_values)
        return df

    def initialize_centroids(self, dataframe) :
        """Initialize random centroids from dataframe"""
        self.centroids = random.sample(dataframe.to_numpy().tolist(), self.get_n_clusters())

    def compare_exact(self, first, second) :
        """Return whether two dicts of numpy arrays are exactly equal"""
        if first.keys() != second.keys():
            return False
        print(all(np.array_equal(first[key], second[key]) for key in first))
        return all(np.array_equal(first[key], second[key]) for key in first)
        

    def fit_kmeans(self, dataframe, max_iter=100) :
        df = dataframe.drop(dataframe.columns[-1], axis=1)
        self.encode_categorical_feature(df)
        self.initialize_centroids(df)
        self.clusters = {k : [] for k in range(self.get_n_clusters())}
        current_clusters = copy.deepcopy(self.clusters)
        for i in range(max_iter) :
            print("iterasi #" + str(i))
            self.clusters = {k : [] for k in range(self.get_n_clusters())}
            current_centroids = copy.deepcopy(self.centroids)
            # print(current_clusters)
            # print(self.clusters)
            for row in df.iterrows() :
                # print(row[1].tolist(), end=" ")
                distances = []
                for centroid in self.centroids :
                    distances.append(self.euclidean_distance(row[1], centroid))
                    # print(row[1].tolist())
                # print(distances, end=" ")
                self.clusters[distances.index(min(distances))].append(row)
            self.update_centroids(df)
            print("current centroid " + str(current_centroids))
            print("self centroid " + str(self.centroids))
            for j in range(self.get_n_clusters()) :
                print("current cluster" + str(j) + " len=" + str(len(current_clusters[j])))
                print("self cluster" + str(j) + " len=" + str(len(self.clusters[j])))
            print()
            if current_centroids == self.centroids or self.compare_exact(current_clusters, self.clusters) :
                print("stop condition satisfied")
                break
            else :
                current_clusters.clear()
                current_clusters = copy.deepcopy(self.clusters)
        print("stopped")
        return

    def update_centroids(self, dataframe) :
        for index in range(self.get_n_clusters()) :
            avg = [0 for i in range(len(dataframe.columns))]
            for data in self.clusters[index] :
                avg += data[1]
            avg /= len(self.clusters[index])
            self.centroids[index] = avg.tolist()
        return