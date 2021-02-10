"""
Neuropixels analysis for the Dacre et al motor thalamus paper.
"""

import os
import sys

sys.path.insert(0, os.path.expanduser("~/git/spikesorters"))
os.environ['KILOSORT3_PATH'] = os.path.expanduser('~/git/Kilosort')
os.environ['KILOSORT2_5_PATH'] = os.path.expanduser('~/git/Kilosort2.5')

import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import clusters, spike_times


mice = [
    #'MCos5',
    #'MCos9',
    #'MCos29',
    'C57_724',
    'C57_1288723',
    'C57_1288727',
]

exp = Experiment(
    mice,
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

#exp.sort_spikes()
#exp.process_behaviour()
#exp.process_lfp()
#exp.process_spikes()
#exp.extract_spikes()
#exp.process_motion_tracking()

#session = 0
#hits = exp.align_trials(ActionLabels.rewarded_push, Events.back_sensor_open, 'spike_times')
#spike_times.across_trials_histogram(hits, session)
#plt.show()

clusters.depth_profile(exp, curated=False)
#plt.ylim([-250, 4000])
plt.show()
