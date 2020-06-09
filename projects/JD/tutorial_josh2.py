"""
This is a tutorial on how to use the various parts of the pixels pipeline.
"""


import matplotlib.pyplot as plt
import seaborn as sns

# from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events, LeverPushExp

# Step 1: Load an experiment
#
# An experiment handles a group of mice that were trained in the same behaviour. It
# stores data and metadata for all included sessions belonging to the list of mice
# provided. The Experiment class is passed the mouse or list of mice, the class
# definition for the behaviour they were trained in (imported from pixels.behaviours),
# and the paths where it can find recording data (the folder containing 'raw', 'interim'
# and 'processed' folders) and training metadata.
#
# myexp = Experiment(
#     'MCos1370',
#     LeverPush,
#     '~/duguidlab/thalamus_paper/Npx_data',
#     '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
# )

myexp2 = LeverPushExp(
    'MCos1370',
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
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
#    - sorted spikes (TODO)
#

# This aligns, crops and downsamples behavioural data.
# myexp.process_behaviour()

# This aligns, crops and downsamples LFP data.
# myexp.process_lfp()

# This aligns, crops, and downsamples spike data.
# myexp.process_spikes()

# This performs spike sorting, and ... (TODO)
# myexp.spike_sort()

# This... (TODO)
# myexp.process_motion_tracking()

# This exracts ITIs
# raw_spike_itis1 = myexp2.extract_ITIs(1, 'spike', raw=False)
# raw_spike_itis2 = myexp2.extract_ITIs(2, 'spike', raw=False)
raw_spike_itis3 = myexp2.extract_ITIs(3, 'spike', raw=False)

# Step 3: Run exploratory analyses
# Once all the data has been processed and converted into forms that are compatible with
# the rest of the data, we are ready to extract data organised by trials.
#
# Data can be loading as trial-aligned data using the Experiment.align_trials method.
# This returns a multidimensional pandas DataFrame containing the desired data organised
# by session, unit, and trial.
#
# Here are some examples of how data can be plotted:
#

# Plotting ds spike itis
num_chan = 2
num_trials = 21
session = 0

fig, axes = plt.subplots(num_chan, 1, sharex=True, figsize=(18, 12))
for trial in range(num_trials):
    for i in range(num_chan):
        sns.lineplot(
            data=raw_spike_itis3[session][1 + i][trial] + 100*trial,
            estimator=None,
            style=None,
            ax=axes[i]
        )

plt.show()

# Extract plottable data
# sample_rate=myexp.sessions[0].sample_rate
allbehavdata=myexp[0].get_behavioural_data()
alllfpdata=myexp[0].get_lfp_data()
allspikedata=myexp.sessions[0].get_spike_data()