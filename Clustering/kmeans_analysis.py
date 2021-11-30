import kmeans 
import matplotlib.pyplot as plt  
import pandas as pd 
import numpy as np 
import random
import sys

cluster = [2, 4, 8, 16, 32]
seed = [2*i for i in range(10)]


def visualize(data, name, k): 
    np.random.seed(0)
    wc_ssd, sc, nmi, result = kmeans.kmeans(data, k)
    print(name + " NMI: " + str(nmi)) 
    
    data = np.append(data, result.reshape(len(result), 1), 1) 
    index = np.random.randint(0, len(data), size=1000)
    picked = data[index, :]
   
    fig, ax = plt.subplots() 
    ax.set_title(name) 
    for i in range(k): 
        group = picked[picked[:, 4] == i]
        plt.scatter(group[:, 2], group[:, 3], label=str(i))
    plt.legend() 
    plt.savefig(name + "_plotCluster.png")


def plotRandom(wc_random, sc_random, name): 
    # wc_random
    fig, ax = plt.subplots() 
    ax.set_title(name + " WC_SSD vs random seed") 
    for i in range(len(cluster)): 
        plt.plot(seed, wc_random[i], label = "k=" + str(cluster[i]))
    plt.xlabel("Random Seed") 
    plt.ylabel("WC_SSD") 
    plt.legend()
    plt.savefig(name + "_wc_random.png")  

    # sc_random
    fig, ax = plt.subplots() 
    ax.set_title(name + " SC vs random seed") 
    for i in range(len(cluster)): 
        plt.plot(seed, sc_random[i], label = "k=" + str(cluster[i]))
    plt.xlabel("Random Seed") 
    plt.ylabel("SC") 
    plt.legend()
    plt.savefig(name + "_sc_random.png") 

def randomSeed(data, name):  
    wc_random = np.zeros((5, 10))
    sc_random = np.zeros((5, 10))
    print(name) 
    index = 0
    for k in cluster: 
        print("Cluster k = " + str(k))
        for i in seed: 
            np.random.seed(i)
            wc_ssd, sc, nmi, result = kmeans.kmeans(data, k)
            wc_random[cluster.index(k), seed.index(i)] = wc_ssd
            sc_random[cluster.index(k), seed.index(i)] = sc 
        print("wc_ssd: avg = {:.3f}, std = {:.3f}".format(np.mean(wc_random[cluster.index(k)]), np.std(wc_random[cluster.index(k)])))
        print("sc: avg = {:.3f}, std = {}".format(np.mean(sc_random[cluster.index(k)]), np.std(sc_random[cluster.index(k)])))
        
    plotRandom(wc_random, sc_random, name)

def compareK(data, name): 
    wc_ssd_list = [] 
    sc_list = []
    for k in cluster: 
        wc_ssd, sc, nmi, result = kmeans.kmeans(data, k)
        wc_ssd_list.append(wc_ssd) 
        sc_list.append(sc)
    
    plotSCSSD(wc_ssd_list, sc_list, cluster, name)

def plotSCSSD(wc_ssd_list, sc_list, cluster, data): 

    fig, ax = plt.subplots() 
    ax.set_title(data + " WC_SSD") 
    ax.plot(cluster, wc_ssd_list)
    plt.xlabel("K") 
    plt.ylabel("WC_SSD") 
    plt.savefig(data + "_wc_ssd.png") 

    # plot sc
    fig, ax = plt.subplots() 
    ax.set_title(data + " SC") 
    ax.plot(cluster, sc_list)
    plt.xlabel("K") 
    plt.ylabel("SC") 
    plt.savefig(data + "_sc.png")
    

if __name__ == "__main__": 
    if len(sys.argv) < 2: 
        print("Missing Argument")
    else: 
        analysis = int(sys.argv[1])
 
    full_data = pd.read_csv("digits-embedding.csv").to_numpy()
    labels = full_data[:, 1] 
    fourDigits = full_data[np.where((labels == 2)| (labels == 4) | (labels == 6) | (labels == 7))[0], :]
    twoDigits = full_data[np.where((labels == 6)| (labels == 7))[0], :]
    
    if analysis == 1:
        print("Compare the impact of various k for each dataset") 
        np.random.seed(0)
        compareK(full_data, "Dataset1")
        compareK(fourDigits, "Dataset2")
        compareK(twoDigits, "Dataset3")
    elif analysis == 2: 
        print("Run analysis with 10 random seed")
        randomSeed(full_data, "Dataset1")
        randomSeed(fourDigits, "Dataset2")
        randomSeed(twoDigits, "Dataset3")
    elif analysis == 3: 
        print("Run analysis with k chosen from 2.2.2")
        visualize(full_data, "Dataset1", 8)
        visualize(fourDigits, "Dataset2", 4)
        visualize(twoDigits, "Dataset3", 2)
