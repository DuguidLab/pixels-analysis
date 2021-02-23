import os

os.environ['KILOSORT3_PATH'] = '/opt/neuropixels/Kilosort'

import matplotlib.pyplot as plt

from pixels import Experiment
from pixels.behaviours.reach import Reach, ActionLabels, Events
from pixtools import clusters, spike_times


mice = [
    "VR16",
    "VR17",
    "VR18",
]

exp = Experiment(
    mice,
    Reach,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)


exp.sort_spikes()
#exp.process_spikes()
#exp.process_lfp()
#exp.process_behaviour()
