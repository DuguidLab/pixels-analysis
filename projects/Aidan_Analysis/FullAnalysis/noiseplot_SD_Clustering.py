# Unlike other noiseplot file (by channel) this file will cluster the SDs, allowing for a mean square analysis
# If there are distinct clusters able to be seperated by depth, it will indicate that there is a clear relationship to noise
# TODO: Test Functionality after noise analysis is complete, push to pixtools.

# First import required packages
import sys
import json

sys.path.insert(
    0, "/home/s1735718/PixelsAnalysis/pixels"
)  # Line will allow me to debug base.py code locally rather than github copy
sys.path.append("/home/s1735718/PixelsAnalysis/pixels-analysis")

from turtle import fd
from base import *
from channeldepth import *
from channeldepth import *
from sklearn.cluster import KMeans

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush
from pixels.behaviours.pushpull import PushPull
from pixels.behaviours.reach import Reach
from pixels.behaviours.no_behaviour import NoBehaviour

import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import matplotlib.pyplot as plt

from pixtools import clusters
from pixtools import utils

# Now we shall read in the experimental data, defined below
# Remember, this selection is defined as exps, NOT myexp which is defined in the normal base file. Check this before continuing!

# First define the experiment to analyse and sessions
noise_tests = Experiment(
    ["noisetest1"],
    NoBehaviour,
    "~/duguidlab/visuomotor_control/neuropixels",
)

noise_test_unchanged = Experiment(
    ["noisetest_unchanged"],
    NoBehaviour,
    "~/duguidlab/visuomotor_control/neuropixels",
)

noise_test_nopi = Experiment(
    ["noisetest_nopi"],
    NoBehaviour,
    "~/duguidlab/visuomotor_control/neuropixels",
)

noise_test_no_caps = Experiment(
    "VR50",
    Reach,
    "~/duguidlab/visuomotor_control/neuropixels",
)

# This contains the list of experiments we want to plot noise for, here only interested in the reaching task
# Remember to change this in base.py
exps = {"reaching": myexp}


def noise_per_channeldepth(exp_list, myexp):
    """
    Function takes the experiments defined above (as exps dictionary) and extracts the noise for each channel, combining this into a dataframe

    exp_list: the experiment dictionary defined above.

    myexp: the experiment defined in base.py, will extract the depth information from here.
    """
    noise = []  # Create the empty array to hold the noise information

    for name, exp in exp_list.items():
        for session in exp:
            for i in range(len(session.files)):
                path = session.processed / f"noise_{i}.json"
                with path.open() as fd:
                    ses_noise = json.load(fd)
                date = datetime.datetime.strptime(session.name[:6], "%y%m%d")
                for SD in ses_noise["SDs"]:
                    noise.append((session.name, date, name, SD))

    df = pd.DataFrame(noise, columns=["session", "date", "project", "SDs"])

    # Now that we have saved the noise for this experiment, must import the depth information and concatenate
    depth = meta_spikeglx(myexp, 0)
    depth = depth.to_dataframe()

    df2 = pd.concat([df, depth], axis=1)

    return df2


# Now that we have both noise and depth information for every channel, we may perform a means clustering
# Initiate the kmeans class, describing the parameters of our analysis
# First double check how many clusters are required to best describe the data
# Reshape the array to allow clustering
df2 = noise_per_channeldepth(exps, myexp)

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


# elbowplot(df2, myexp)
# Must now define kmeans parameters based on elbow analysis!
# Seems that 2 clusters is still optimal across datasets

kmeans = KMeans(
    init="random",  # Initiate the iterative analysis with random centres
    n_clusters=2,  # How many clusters to bin the data into, based on the elbow analysis!
    n_init=10,  # Number of centroids to generate initially
    max_iter=300,  # Max number of iterations before ceasing analysis
    random_state=42,  # The random number seed for centroid generation, can really be anything for our purposes
)


for s, session in enumerate(myexp):
    name = session.name

    df2 = df2.loc[data["session"] == name]
    x_kmeans = kmeans.fit(df2)
    y_means = kmeans.fit_predict(df2)

    # Now plot the kmeans analysis
    # Remember we use our original data (df2) but use the df3 analysis to generate the labels
    plt.scatter(df2["y"], df2["SDs"], c=y_means, cmap="viridis")
    plt.xlabel("Probe Channel Y-Coordinate")
    plt.ylabel("Channel Noise (SD)")
    plt.title(f"{myexp[0].name} Channel Noise k-Mean Clustering Analysis")

    plt.show()
