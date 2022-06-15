# Unlike other noiseplot file (by channel) this file will cluster the SDs, allowing for a mean square analysis
# If there are distinct clusters able to be seperated by depth, it will indicate that there is a clear relationship to noise

# First import required packages
import sys
import json

from turtle import fd
from channeldepth import *
from channeldepth import *
from sklearn.cluster import KMeans

import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import matplotlib.pyplot as plt


def noise_per_channeldepth(myexp):
    """
    Function  extracts the noise for each channel, combining this into a dataframe

    myexp: the experiment defined in base.py, will extract the depth information from here.
    """
    noise = pd.DataFrame(
        columns=["session", "project", "SDs", "x", "y"]
    )  # Create the empty array to hold the noise information
    depths = meta_spikeglx(myexp, 0)
    depths = depths.to_dataframe()
    coords = depths[
        ["x", "y"]
    ]  # Create a dataframe containing the generic x and y coords.
    tot_noise = []

    # Iterate through each session, taking the noise for each file and loading them into one continuous data frame.
    for s, session in enumerate(myexp):
        for i in range(len(session.files)):
            path = session.processed / f"noise_{i}.json"
            with path.open() as fd:
                ses_noise = json.load(fd)

            chan_noises = []
            for j, SD in enumerate(
                ses_noise["SDs"][0:-1]
            ):  # This will iterate over first 384 channels, and exclude the sync channel
                x = coords["x"].iloc[j]
                y = coords["y"].iloc[j]
                noise_row = pd.DataFrame.from_records(
                    {"session": [session.name], "SDs": [SD], "x": x, "y": y}
                )
                chan_noises.append(noise_row)

        # Take all datafrom channel noises for a session, then concatenate
        noise = pd.concat(chan_noises)
        tot_noise.append(noise)  # Take all channel noises and add to a master file
        df2 = pd.concat(
            tot_noise
        )  # Convert this master file, containing every sessions noise data into a dataframe

    return df2


# Now determine the optimal number of clusters to use in the K-means analysis by producing elbow plots
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


def clusterplot(data, myexp, cluster_num):
    """
    Function takes the noise per channel and depth information, produced by noise_per_channeldepth() and produces a clusterplot.
    Clustering is performed by K-means analysis, elbow plot should be produced by elbowplot() to determine optimal cluster number.

    data: data produced by noise_per_channel_depth() containing channel ID, coordinate, and recording noise for each session in the exp class

    myexp: the exp class containing mouse IDs

    cluster_num: the number of clusters to produce through the k-means analysis, determined by qualitative analysis of elbow plots. (where the "bow" of the line occurs)

    """

    # First define k-means parameters for clustering
    kmeans = KMeans(
        init="random",  # Initiate the iterative analysis with random centres
        n_clusters=cluster_num,  # How many clusters to bin the data into, based on the elbow analysis!
        n_init=10,  # Number of centroids to generate initially
        max_iter=300,  # Max number of iterations before ceasing analysis
        random_state=42,  # The random number seed for centroid generation, can really be anything for our purposes
    )
    
    df2 = data
    for s, session in enumerate(myexp):
        name = session.name

        ses = df2.loc[df2["session"] == name]
        sd = ses["SDs"].values.reshape(-1, 1)
        y_means = kmeans.fit_predict(sd)

        # Now plot the kmeans analysis
        # Remember we use our original data (df2) but use the df3 analysis to generate the labels
        plt.scatter(ses["y"], ses["SDs"], c=y_means, cmap="viridis")

        plt.xlabel("Probe Channel Y-Coordinate")
        plt.ylabel("Channel Noise (SD)")
        plt.title(f"{name} Channel Noise k-Mean Clustering Analysis")

        plt.show()
