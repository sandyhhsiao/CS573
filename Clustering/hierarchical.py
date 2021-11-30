import sys 
import random
import pandas as pd
import numpy as np
import kmeans 
import kmeans_analysis
import matplotlib.pyplot as plt 
import scipy.cluster.hierarchy as hac

kValues = [2, 4, 8, 16, 32]
def clusterData(data, method): 
    result = hac.linkage(data[:, 2:], method = method) 
    plt.figure()
    graph = hac.dendrogram(result) 
    plt.savefig("dendrogram_"+method+".png")

    wc_ssd_list = [] 
    sc_list = []
    nmi_list = []
    for k in kValues: 
        cluster_dict = {}
        centroids = {}
        cluster = hac.fcluster(result, k, criterion="maxclust") 
        cluster -= 1 #cluster result starts from 1 
        for i in range(k): 
            examples = data[np.where(cluster == i)[0], 2:]
            if len(examples) != 0:
                cluster_dict[i] = examples
                centroids[i] = np.mean(cluster_dict[i], axis=0)

        wc_ssd_list.append(kmeans.computeSSD(cluster_dict, centroids, k))
        sc_list.append(kmeans.computeSC(data, cluster, k))
        nmi_list.append(kmeans.computeNMI(data, cluster, k))
    kmeans_analysis.plotSCSSD(wc_ssd_list, sc_list, kValues, method+"-linkage")
    print(method + " NMI: ")
    print(nmi_list)

if __name__ == "__main__":
    embed_data = pd.read_csv("digits-embedding.csv").to_numpy()
    
    # select 10 image from each digit
    random.seed(0)
    picked = []
    for i in range(10): 
        group = np.where(embed_data[:, 1] == i)[0] 
        index = random.choices(group, k=10)
        picked = picked + index
    
    data = embed_data[picked, :]
    clusterData(data, "single")
    clusterData(data, "complete")
    clusterData(data, "average")
