import sys 
import pandas as pd
import numpy as np 
import random
from scipy.spatial import distance

def computeNMI(data, cluster, k):
    classLabel = np.unique(data[:, 1]) 
    classDict = {label : 0.0 for label in classLabel}
    clusterDict = {c : 0.0 for c in range(k)}
    data_size = len(cluster)
    #compute H(C)
    hClass = 0
    for i in classLabel: 
        classDict[i] = len(np.where(data[:, 1] == i))/data_size
        hClass -= classDict[i]*np.log(classDict[i])

    #compute H(G)
    hCluster = 0
    for i in range(k): 
        clusterDict[i] = len(np.where(cluster == i))/data_size
        hCluster -= clusterDict[i]*np.log(clusterDict[i])

    #compute information gain 
    #I(C, G) = H(C) - H(C|G)
    
    #H(C|G)
    hClassCluster = 0
    for g in range(k):
        examples = data[np.where(cluster == g)[0], :]
        tmp = 0
        for c in classLabel:  
            points = np.where(examples[:, 1] == c)[0]
            if len(points) != 0: 
                prob = len(points)/len(examples)
                tmp += prob*np.log(prob)
        hClassCluster -= clusterDict[g]*tmp 
    return (hClass - hClassCluster)/(hClass + hCluster)

def computeSC(data, cluster, k):
    sc = []
    dist = distance.cdist(data[:, 2:], data[:, 2:], 'euclidean')
    for i in range(len(cluster)):
        SA = 0
        SB = -1
        for j in range(k): 
            avgDist = np.average(dist[i][np.where(cluster == j)[0]]) 
            if j == cluster[i]: 
                SA = avgDist 
            else: 
                SB = avgDist if SB == -1 else min(avgDist, SB)
        sc.append((SB - SA)/max(SA, SB))
    return np.average(sc)

def computeSSD(data_cluster, centroids, k): 
    wc_ssd = 0
    for i in range(k): 
        wc_ssd += np.sum(np.square(data_cluster[i] - centroids[i]))
    return wc_ssd

def kmeans(data, k):
    index = np.random.choice(len(data), k, replace=False) 
    centroids = data[index, 2:]
    print(centroids)
    data_cluster = {}
    cluster = []
    iteration = 0
    while iteration < 50:
        dist = distance.cdist(data[:, 2:], centroids, 'euclidean')
        cluster = np.argmin(dist, axis=1)
        iteration += 1

        #update centroids
        if iteration != 50:
            update = []
            for i in range(k): 
                data_cluster[i] = data[np.where(cluster == i)[0], 2:]
                update.append(np.mean(data_cluster[i], axis=0))
        
            if np.array_equal(centroids, update):
                break
            else:
                centroids = update

    wc_ssd = computeSSD(data_cluster, centroids, k)
    sc = computeSC(data, cluster, k)
    nmi = computeNMI(data, cluster, k)
    return wc_ssd, sc, nmi, cluster
    

if __name__ == "__main__": 
    if len(sys.argv) < 3: 
        print("Missing arguments")
    else: 
        data = pd.read_csv(sys.argv[1]).to_numpy()
        cluster = int(sys.argv[2])
        np.random.seed(0)
        kmeans(data, cluster)
        