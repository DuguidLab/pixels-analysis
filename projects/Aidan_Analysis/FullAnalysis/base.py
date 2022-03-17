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
from pixels import Experiment
from pixels.behaviours.reach import Reach, ActionLabels, Events
from pixtools import utils

# Step 1: Load an experiment
#
# An experiment handles a group of mice that were trained in the same behaviour. It
# stores data and metadata for all included sessions belonging to the list of mice
# provided. The Experiment class is passed the mouse or list of mice, the class
# definition for the behaviour they were trained in (imported from pixels.behaviours),
# and the paths where it can find recording data (the folder containing 'raw', 'interim'
# and 'processed' folders) and training metadata.

# Will import master data from visuomotor control/neuropixels

# Import the newly recorded VR59 data, there are three sessions (over three days of recording)!
myexp = Experiment(
    ["VR46", "VR47", "VR49", "VR52",  "VR55", "VR59"],  # This can be a list
    Reach,  # We want the reach behaviour
    "~/duguidlab/visuomotor_control/neuropixels",  # Where is the main data saved
    "~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON",  # Where is the metadata for the recording saved
)
