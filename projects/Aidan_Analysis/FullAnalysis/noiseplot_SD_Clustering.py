# Unlike other noiseplot file (by channel) this file will cluster the SDs, allowing for a mean square analysis
# If there are distinct clusters able to be seperated by depth, it will indicate that there is a clear relationship to noise
# TODO: Test Functionality after noise analysis is complete, push to pixtools.
# First import required packages

import json
import sys

sys.path.insert(
    0, "/home/s1735718/PixelsAnalysis/pixels"
)  # Line will allow me to debug base.py code locally rather than github copy
sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis")

import datetime
from turtle import fd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush
from pixels.behaviours.no_behaviour import NoBehaviour
from pixels.behaviours.pushpull import PushPull
from pixels.behaviours.reach import Reach
from pixtools import clusters, utils
from sklearn.cluster import KMeans

from base import *
from channeldepth import *


#TODO: Fix this function! Why isn't it properly concatenating.
def noise_per_channeldepth(myexp):
    """
    Function  extracts the noise for each channel, combining this into a dataframe

    myexp: the experiment defined in base.py, will extract the depth information from here.
    """
    noise = pd.DataFrame(columns=["session", "project", "SDs", "x", "y"])  # Create the empty array to hold the noise information
    depths = meta_spikeglx(myexp, 0) 
    depths = depths.to_dataframe() 
    coords = depths[["x", "y"]] # Create a dataframe containing the generic x and y coords. 
    tot_noise = []

    #Iterate through each session, taking the noise for each file and loading them into one continuous data frame.
    for s, session in enumerate(myexp):
        for i in range(len(session.files)):
            path = session.processed / f"noise_{i}.json"
            with path.open() as fd:
                ses_noise = json.load(fd)
            
            chan_noises = []
            for j, SD in enumerate(ses_noise["SDs"][0:-1]): #This will iterate over first 384 channels, and exclude the sync channel
                x = coords["x"].iloc[j]
                y = coords["y"].iloc[j]
                noise_row = pd.DataFrame.from_records(
                    {"session":[session.name], "SDs":[SD], "x": x, "y": y}
                )
                chan_noises.append(noise_row)

        #Take all datafrom channel noises for a session, then concatenate
        noise = pd.concat(chan_noises)
        tot_noise.append(noise) #Take all channel noises and add to a master file
        df2 = pd.concat(tot_noise) #Convert this master file, containing every sessions noise data into a dataframe
    
    return df2

df2 = noise_per_channeldepth(myexp)

# Now that we have both noise and depth information for every channel, we may perform a means clustering
# Initiate the kmeans class, describing the parameters of our analysis
# First double check how many clusters are required to best describe the data
# Reshape the array to allow clustering


# Now determine the optimal number of clusters
def elbowplot(data, myexp):

    """

    This function takes data formatted according to the function above, containing the noise values for all channels
    Will iterate through each experimental session, producing the appropriate graph. Should take the optimal number of clusters as the point at which the elbow bends.
    This point is defined as the boundary where additional clusters no longer explain much more variance in the data.

    data: The dataframe, as formatted by noise_per_channel()

    myexp: The experiment, defined in base.py containing the session information.

    """

    for s, session in enumerate(myexp):
        name = session.name
        ses_data = data.loc[data["session"] == name]
        df3 = ses_data["SDs"].values.reshape(
            -1, 1
        )  # Just gives all noise values, for each session
        Sum_of_squares = []  # create an empty list to store these in.

        k = range(1, 10)
        for num_clusters in k:
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(df3)
            Sum_of_squares.append(kmeans.inertia_)

        fig, ax = plt.subplots()

        # This code will plot the elbow graph to give an overview of the variance in the data explained by the varying the number of clusters
        # This gives the distance from the centroids, as a measure of the variability explained
        # We want this to drop off indicating that there is no remaining data explained by further centroid inclusion

        # Figure has two rows, one columns, this is the first plot
        plt.plot(k, Sum_of_squares, "bx-")  # bx gives blue x as each point.
        plt.xlabel("Putative Number of Clusters")
        plt.ylabel("Sum of Squares Distances/Inertia")
        plt.title(
            f"Determining Optimal Number of Clusters for Analysis - Session {name}"
        )

        plt.show()


#elbowplot(df2, myexp)

# Must now define kmeans parameters based on elbow analysis!
# Seems that 2 clusters is still optimal across datasets

kmeans = KMeans(
    init="random",  # Initiate the iterative analysis with random centres
    n_clusters=2,  # How many clusters to bin the data into, based on the elbow analysis!
    n_init=10,  # Number of centroids to generate initially
    max_iter=300,  # Max number of iterations before ceasing analysis
    random_state=42,  # The random number seed for centroid generation, can really be anything for our purposes
)

# Plot the kmeans clustering by depth (y coord) with hue representing generated clusters. 
for s, session in enumerate(myexp):
    name = session.name

    ses = df2.loc[df2["session"] == name]
    df3 = ses["SDs"].values.reshape(-1, 1)
    x_kmeans = kmeans.fit(df3)
    y_means = kmeans.fit_predict(df3)

    # Now plot the kmeans analysis
    # Remember we use our original data (df2) but use the df3 analysis to generate the labels
    plt.scatter(ses["y"], ses["SDs"], c=y_means, cmap = "viridis"

    )
    
    plt.xlabel("Probe Channel Y-Coordinate")
    plt.ylabel("Channel Noise (SD)")
    plt.title(f"{myexp[s].name} Channel Noise k-Mean Clustering Analysis")

    #Save figures to folder
    utils.save(f"/home/s1735718/PixelsAnalysis/pixels-analysis/projects/Aidan_Analysis/FullAnalysis/Figures/{myexp[s].name}_noise_clustering")
    plt.show()
