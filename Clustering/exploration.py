import sys 
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def plotDigit(data):
    np.random.seed(0)
    fig = plt.figure()
    for i in range(10):
        digit = data[data.iloc[:,1] == i] 
        picked = random.choice(digit.index)
        image = np.array(data.iloc[picked, 2:]).reshape([28,28])
        ax = plt.subplot2grid((2, 5), (int(i/5), i%5))
        ax.imshow(image, cmap='gray')
    fig.savefig("plotDigit.png")

def plotCluster(data): 
    np.random.seed(0)
    fig = plt.figure()
    index = np.random.randint(0, len(data), size=1000)
    #index = random.choices(data.index.tolist(), k=1000)
    picked = data.iloc[index, :]
   
    for i in range(10): 
        group = picked[picked.iloc[:, 1] == i]
        plt.scatter(group.iloc[:, 2], group.iloc[:, 3], label=str(i))
    plt.legend() 
    plt.savefig("plot1000.png")

if __name__ == "__main__":
    raw_data = pd.read_csv("digits-raw.csv") 
    plotDigit(raw_data)

    embed_data = pd.read_csv("digits-embedding.csv") 
    plotCluster(embed_data)

