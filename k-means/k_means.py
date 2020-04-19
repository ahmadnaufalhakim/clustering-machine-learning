import pandas as pd
import numpy as np
import random
import copy
import math
from sklearn.metrics import fowlkes_mallows_score, silhouette_score

random.seed(13517055)

class K_Means :
    def __init__(self):
        self.n_clusters = 0
        self.centroids = []
        self.clusters = {}
        self.true_labels = []
        self.pred_labels = []

    def get_n_clusters(self) :
        """Get number of clusters"""
        return self.n_clusters

    def set_n_clusters(self, n_clusters) :
        """Set number of clusters"""
        self.n_clusters = n_clusters

    def get_true_labels(self) :
        return self.true_labels

    def get_pred_labels(self) :
        return self.pred_labels

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

    def initialize_true_labels(self, dataframe) :
        """Get true labels of data"""
        self.true_labels = [None] * len(dataframe)
        df = dataframe
        if df[df.columns[-1]].dtypes != "int64" and df[df.columns[-1]].dtypes != "float64" :
            feature_values = {}
            for index, value in enumerate(df[df.columns[-1]].unique()) :
                feature_values[value] = index
            for row in df.iterrows() :
                for key in feature_values.keys() :
                    if row[1][-1] == key :
                        self.true_labels[row[0]] = feature_values.get(key)
        else :
            self.true_labels = df[df.columns[-1]].tolist()

    def compare_clusters(self, before, after) :
        """Return whether two dicts of numpy arrays are exactly equal"""
        for key in before.keys() :
            if len(before[key]) != len(after[key]) :
                return False
            for value in range(len(before[key])) :
                if before[key][value][0] != after[key][value][0] :
                    return False
        return True

    def fit_kmeans(self, dataframe, max_iter=100) :
        """K-means fit"""
        # Get all data's true label
        self.initialize_true_labels(dataframe)

        # Dropping the dataframe's target column
        df = dataframe.drop(dataframe.columns[-1], axis=1)

        # Handle categorical features
        self.encode_categorical_features(df)

        # Initialize centroids, clusters, and pred labels
        self.initialize_centroids(df)
        self.clusters = {k : [] for k in range(self.get_n_clusters())}
        self.pred_labels = [None] * len(df)

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
            self.update_pred_labels()

            # Print info
            print("Current centroid " + str(current_centroids))
            print("Self centroid " + str(self.centroids))
            for j in range(self.get_n_clusters()) :
                print("Current cluster " + str(j) + ": len=" + str(len(current_clusters[j])) + " | Self cluster " + str(j) + ": len=" + str(len(self.clusters[j])))
            print("Fowlkes-Mallows score : " + str(fowlkes_mallows_score(self.get_true_labels(), self.get_pred_labels())))
            print("Silhouette Coefficient score : " + str(silhouette_score(df, self.get_pred_labels())))
            print()

            # Check if centroids and clusters are still the same as before
            if current_centroids == self.centroids and self.compare_clusters(current_clusters, self.clusters) :
                print("Stop conditions satisfied")
                break
            else :
                # Re-save current clusters for cluster stop condition checking
                current_clusters.clear()
                current_clusters = copy.deepcopy(self.clusters)
                # Re-initialize current clusters, self.clusters, and data labels
                self.clusters = {k : [] for k in range(self.get_n_clusters())}
                self.pred_labels = [None] * len(dataframe)

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
    
    def update_pred_labels(self) :
        """Update data labels value based on the clusters"""
        for key in self.clusters.keys() :
            for data in range(len(self.clusters[key])) :
                self.pred_labels[self.clusters[key][data][0]] = key