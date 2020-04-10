import pandas as py
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(13517055)

class K_Means :
    def __init__(self):
        self.n_clusters = 0
        self.clusters = np.array([])
    
    def get_n_clusters(self) :
        return self.n_clusters

    def set_n_clusters(self, n_clusters) :
        self.n_clusters = n_clusters

    def euclidean_distance(self, A, B) :
        return np.linalg.norm(A - B)

    def init(self, dataframe) :
        self.clusters = np.array(random.sample(list(dataframe.to_numpy()), self.get_n_clusters()))
        return 