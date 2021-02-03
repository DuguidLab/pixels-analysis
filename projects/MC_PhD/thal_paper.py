"""
This is a tutorial on how to use the various parts of the pixels pipeline.
"""


import os
import sys
sys.path.insert(0, os.path.expanduser("~/git/spikesorters"))
os.environ['KILOSORT3_PATH'] = os.path.expanduser('~/git/Kilosort')


import matplotlib.pyplot as plt
from spikesorters import Kilosort3Sorter

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import spike_times

mice = [
    #'MCos9',
    #'C57_724',  # needs doing with KS3
    'C57_1288723',  # done
    #'C57_1288727',  # needs doing with KS3
]

exp = Experiment(
    mice,
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

#exp.sort_spikes()
exp.process_behaviour()
#exp.process_lfp()
#exp.process_spikes()
#exp.extract_spikes()
#exp.process_motion_tracking()

session = 0
hits = exp.align_trials(ActionLabels.rewarded_push, Events.back_sensor_open, 'spike_times')
spike_times.across_trials_histogram(hits, session)
plt.show()
