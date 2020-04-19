import pandas as pd
from k_means import K_Means

def main() :
    km = K_Means()
    df = pd.read_csv("../csv/iris.csv")
    km.set_n_clusters(3)
    km.fit_kmeans(df)
    print()
    # print("Centroids results:")
    # print("len=" + str(len(km.centroids)) + "\ncentroids:" + str(km.centroids) + "\n" + str(type(km.centroids)) + "\n")
    # print("len=" + str(len(km.clusters)) + "\nclusters:" + str(km.clusters) + "\n" + str(type(km.clusters)) + "\n")
    # print("len=" + str(len(km.data_labels)) + "\nclusters_labels:" + str(km.data_labels) + "\n" + str(type(km.data_labels)))

if __name__ == "__main__":
    main()