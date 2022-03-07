#Unlike other noiseplot file (by channel) this file will cluster the SDs, allowing for a mean square analysis
#If there are distinct clusters able to be seperated by depth, it will indicate that there is a clear relationship to noise

#First import required packages
import sys

sys.path.insert(0, "/home/s1735718/PixelsAnalysis/pixels") #Line will allow me to debug base.py code locally rather than github copy
sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis")

from turtle import fd

from matplotlib.pyplot import suptitle, xlabel
from base import *
from noisebase import *
from channeldepth import *
from channeldepth import *
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd
import seaborn as sns


from pixtools import clusters

#Now we shall read in the experimental data, here this shall be VR59 as specified in noisebase
#Remember, this selection is defined as exps, NOT myexp which is defined in the normal base file. Check this before continuing!
#Could I convert this to a function after refining it??

#Import the results of the noise analysis to a dataframe. 
noise = [] #Create the empty array to hold the noise information

for name, exp in exps.items():
    for session in exp:
        for i in range(len(session.files)):
            path = session.processed / f"noise_{i}.json"
            with path.open() as fd:
                ses_noise = json.load(fd)
            date = datetime.datetime.strptime(session.name[:6], "%y%m%d")
            for SD in ses_noise["SDs"]:
                noise.append((session.name, date, name, SD))

df = pd.DataFrame(noise, columns=["session", "date", "project", "SDs"])

#Now that we have saved the noise for this experiment, must import the depth information and concatenate
depth = meta_spikeglx(myexp, 0)
depth = depth.to_dataframe()

df2 = pd.concat([df, depth], axis=1)

#Now that we have both noise and depth information for every channel, we may perform a means clustering
#Initiate the kmeans class, describing the parameters of our analysis
#First double check how many clusters are required to best describe the data
#Reshape the array to allow clustering
df3=df2["SDs"].values.reshape(-1, 1)
Sum_of_squares=[]

#Now determine the optimal number of clusters
k=range(1,10)
for num_clusters in k:
    kmeans=KMeans(n_clusters=num_clusters)
    kmeans.fit(df3)
    Sum_of_squares.append(kmeans.inertia_)

fig, ax = plt.subplots(2,1)

#This code will plot the elbow graph to give an overview of the variance in the data explained by the varying the number of clusters
#This gives the distance from the centroids, as a measure of the variability explained
#We want this to drop off indicating that there is no remaining data explained by further centroid inclusion

plt.subplot(2, 1, 1) #Figure has two rows, one columns, this is the first plot
plt.plot(k, Sum_of_squares, "bx-") #bx gives blue x as each point.
plt.xlabel("Putative Number of Clusters")
plt.ylabel("Sum of Squares Distances/Inertia")
plt.title("Determining Optimal Number of Clusters for Analysis")


#Must define kmeans parameters based on elbow analysis!
kmeans = KMeans(
    init="random", #Initiate the iterative analysis with random centres 
    n_clusters=2, #How many clusters to bin the data into, based on the elbow analysis
    n_init=10, #Number of centroids to generate initially
    max_iter=300, #Max number of iterations before ceasing analysis
    random_state=42 #The random number seed for centroid generation
)

x_kmeans=kmeans.fit(df3)
y_means=kmeans.fit_predict(df3)

#Now plot the kmeans analysis
#Remember we use our original data (df2) but use the df3 analysis to generate the labels
plt.subplot(2, 1, 2)
plt.scatter(df2["y"], df2["SDs"], c=y_means, cmap="viridis")
plt.xlabel("Probe Channel Y-Coordinate")
plt.ylabel("Channel Noise (SD)")
plt.title(f"{myexp[0].name} Channel Noise k-Mean Clustering Analysis")

plt.show()