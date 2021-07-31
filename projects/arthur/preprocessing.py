"""
Pre-processing reach npx data.
"""

from pixels import Experiment
from pixels.behaviours.reach import Reach, ActionLabels, Events


# Step 1: Load an experiment
#
# An experiment handles a group of mice that were trained in the same behaviour. It
# stores data and metadata for all included sessions belonging to the list of mice
# provided. The Experiment class is passed the mouse or list of mice, the class
# definition for the behaviour they were trained in (imported from pixels.behaviours),
# and the paths where it can find recording data (the folder containing 'raw', 'interim'
# and 'processed' folders) and training metadata.
#
myexp = Experiment(
    'HFR25',  # This can be a list
    Reach,
    '/home/s2120426/duguidlab/visuomotor_control/neuropixels',
    '/home/s2120426/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)


# Step 2: Process raw data
#
# These methods each process a different type of raw data and save the output into the
# 'processed' folder. The outputs are all resampled to 1kHz, and are saved as:
#
#    - action labels (.npy)
#    - behavioural data (.h5)
#    - LFP data (.h5)
#    - spike data (.h5)
#    - sorted spikes ()
#

# This aligns, crops and downsamples behavioural data.
myexp[1].process_behaviour()

# This aligns, crops and downsamples LFP data. Currently unused.
#myexp.process_lfp()

# This aligns, crops, and downsamples spike data. Currently unused.
#myexp[0].process_spikes()

# This runs the spike sorting algorithm and outputs the results in a form usable by phy.
myexp[1].sort_spikes()

# This extracts posture coordinates from TDMS videos using DeepLabCut
#config = '/path/to/this/behaviours/deeplabcut/config.yaml'
#myexp.process_motion_tracking(config)

# If you also want to output labelled videos, pass this keyword arg:
#myexp.process_motion_tracking(config, create_labelled_video=True):

# This method will convert the videos from TDMS to AVI before running them through DLC.
# If you just want the AVI videos without the DLC, you can do so directly:
#myexp.extract_videos()


# Step 3: Run exploratory analyses
# see "tutorial.py"
