import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from k_means import K_Means
import random

def main() :
    km = K_Means()
    df = pd.read_csv("../csv/iris.csv")
    km.set_n_clusters(3)
    km.fit_kmeans(df)
    print()
    print("Centroids results:")
    print("len=" + str(len(km.centroids)) + "\ncentroids:" + str(km.centroids) + "\n" + str(type(km.centroids)) + "\n")
    print("len=" + str(len(km.clusters)) + "\nclusters:" + str(km.clusters) + "\n" + str(type(km.clusters)) + "\n")
    print("len=" + str(len(km.clusters_labels)) + "\nclusters_labels:" + str(km.clusters_labels) + "\n" + str(type(km.clusters_labels)))

if __name__ == "__main__":
    main()