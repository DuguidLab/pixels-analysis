# This file sets up the required data/paths for subsequent data processing
# TODO: add pixtools to path permenantly so I don't have to keep importing from a local copy
# TODO: run files through black to format

# Package required to change path
from pathlib import Path

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fig_dir = Path(
    "/home/s1735718/PixelsAnalysis/pixels-analysis/projects/Aidan_Analysis/FullAnalysis/Figures"
)

sys.path.append(
    "/home/s1735718/PixelsAnalysis/pixels-analysis"
)  # Adds the location of the pixtools folder to the python path
# sys.path.insert(0, "/home/s1735718/PixelsAnalysis/pixels")
from pixels import Experiment
from pixels.behaviours.reach import Reach, ActionLabels, Events
from pixtools import utils

interim_dir = "/data/visuomotor_control/interim"

# Step 1: Load an experiment
#
# An experiment handles a group of mice that were trained in the same behaviour. It
# stores data and metadata for all included sessions belonging to the list of mice
# provided. The Experiment class is passed the mouse or list of mice, the class
# definition for the behaviour they were trained in (imported from pixels.behaviours),
# and the paths where it can find recording data (the folder containing 'raw', 'interim'
# and 'processed' folders) and training metadata.

# Will import master data from visuomotor control/neuropixels

myexp = Experiment(
    [
        "VR46",
        # "VR47",
        # "VR49",
        # "VR50",
        # "VR52",
        # "VR53",
        # "VR54",
        # "VR55",
        # "VR56",
        # "VR58"
        # "VR59",
    ],  # This can be a list
    Reach,  # We want the reach behaviour
    "~/duguidlab/visuomotor_control/neuropixels",  # Where is the main data saved
    "~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON",  # Where is the metadata for the recording saved
    interim_dir="/data/visuomotor_control/interim",
)


# This depth range covers all mice
m2_depth = {
    "min_depth": 200,
    "max_depth": 1200,
}


def get_pyramidals(exp):
    rep = "-".join(str(s) for s in m2_depth.values())

    return exp.select_units(
        **m2_depth,
        min_spike_width=0.4,
        name=f"{rep}_pyramidals",
    )


def get_interneurons(exp):
    rep = "-".join(str(s) for s in m2_depth.values())

    return exp.select_units(
        **m2_depth,
        max_spike_width=0.35,
        name=f"{rep}_interneurons",
    )
