import pandas as pd
import numpy as np
import random
import copy
import math

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

    def encode_categorical_features(self, dataframe) :
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

    def compare_clusters(self, before, after) :
        """Return whether two dicts of numpy arrays are exactly equal"""
        for key in before.keys() :
            if len(before[key]) != len(after[key]) :
                return False
            for value in range(len(before[key])) :
                # print(before[key][value][0] == after[key][value][0])
                if before[key][value][0] != after[key][value][0] :
                    return False
        return True

    def fit_kmeans(self, dataframe, max_iter=100) :
        """K-means fit"""
        # Dropping the dataframe's target column
        df = dataframe.drop(dataframe.columns[-1], axis=1)
        
        # Handle categorical features
        self.encode_categorical_features(df)

        # Initialize centroids and clusters
        self.initialize_centroids(df)
        self.clusters = {k : [] for k in range(self.get_n_clusters())}

        # Save current clusters for cluster stop condition checking
        current_clusters = copy.deepcopy(self.clusters)
        
        for i in range(max_iter) :
            print("Iteration #" + str(i+1))

            # Save current centroids for centroid stop condition checking
            current_centroids = copy.deepcopy(self.centroids)

            # Iterate through all dataframe rows
            for row in df.iterrows() :
                
                # Store distances from data row to all centroids
                distances = []
                
                # Iterate through all centroids
                for centroid in self.centroids :
                    distances.append(self.euclidean_distance(row[1], centroid))
                
                # Assign rows to the nearest cluster
                self.clusters[distances.index(min(distances))].append(row)
            
            # Update centroids based on the corresponding clusters
            self.update_centroids(df)

            print("current centroid " + str(current_centroids))
            print("self centroid " + str(self.centroids))
            for j in range(self.get_n_clusters()) :
                print("current cluster" + str(j) + " len=" + str(len(current_clusters[j])))
                print("self cluster" + str(j) + " len=" + str(len(self.clusters[j])))
            print()

            # Check if centroids and clusters are still the same as before
            if current_centroids == self.centroids and self.compare_clusters(current_clusters, self.clusters) :
                print("stop condition satisfied")
                break
            else :
                current_clusters.clear()
                current_clusters = copy.deepcopy(self.clusters)
                self.clusters = {k : [] for k in range(self.get_n_clusters())}
        
        print("clustering stopped")
        return

    def update_centroids(self, dataframe) :
        """Update centroids value based on the corresponding clusters"""
        for index in range(self.get_n_clusters()) :
            avg = [0 for i in range(len(dataframe.columns))]
            for data in self.clusters[index] :
                avg += data[1]
            avg /= len(self.clusters[index])
            self.centroids[index] = avg.tolist()