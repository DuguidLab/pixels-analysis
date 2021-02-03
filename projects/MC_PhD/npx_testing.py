import matplotlib.pyplot as plt
import seaborn as sns
from spikesorters import Kilosort2Sorter

from pixels import Experiment
from pixels.behaviours.reach import Reach, ActionLabels, Events


import os
Kilosort2Sorter.set_kilosort2_path(os.path.expanduser('~/git/Kilosort2'))


mice = [
    "VR16",
    #"VR17",
    #"VR18",
]

exp = Experiment(
    mice,
    Reach,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)


#exp.sort_spikes()
exp.process_spikes()
#exp.process_lfp()
#exp.process_behaviour()
