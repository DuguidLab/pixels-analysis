import matplotlib.pyplot as plt
import seaborn as sns
from spikesorters import Kilosort2Sorter

from pixels import Experiment
from pixels.behaviours.reach import Reach, ActionLabels, Events


import os
Kilosort2Sorter.set_kilosort2_path(os.path.expanduser('~/git/Kilosort2'))


mice = [
    "VR16",
]

exp = Experiment(
    mice,
    Reach,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)


exp.sort_spikes()
