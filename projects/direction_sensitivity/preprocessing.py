"""
Neuropixels analysis for the direction sensitivity project.
"""

import os
os.environ['KILOSORT3_PATH'] = '/opt/neuropixels/Kilosort'
os.environ['KILOSORT2_5_PATH'] = os.path.expanduser('~/git/Kilosort2.5')

import matplotlib.pyplot as plt

from pixels import Experiment
from pixels.behaviours.pushpull import PushPull, ActionLabels, Events

mice = [
    #   First cohort:
    #"C57_1319786",
    #"C57_1319781",
    #"C57_1319784",
    #"C57_1319783",
    #"C57_1319782",
    #"C57_1319785",
    #   Second cohort:
    "C57_1350950",
    "C57_1350951",
    "C57_1350952",
    #"C57_1350953",  # corrupted AP data
    "C57_1350954",  # Actions were deleted manually from action labels after 20th push due to bad MI
    "C57_1350955",
]

exp = Experiment(
    mice,
    PushPull,
    '~/duguidlab/Direction_Sensitivity/Data/Neuropixel',
    #'~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

#exp.sort_spikes()
#exp.process_lfp()
#exp.process_spikes()
#exp.extract_videos(force=True)
#exp.process_behaviour()
#exp.process_motion_tracking()
#exp.assess_noise()

## cutting out sample
#self = exp[0]
#rec_num = 0
#recording = self.files[0]
#data_file = self.find_file(recording['spike_data'])
#orig_rate = self.spike_meta[rec_num]['imSampRate']
#num_chans = self.spike_meta[rec_num]['nSavedChans']
#from pixels import ioutils
#data = ioutils.read_bin(data_file, num_chans)
#t = data.shape[0] // 2 #n = 8
#
#new = data[t : t + 30000*300, 17:17+n]
#
##fig, axes = plt.subplots(n, 1, sharex=True, sharey=True)
##for i in range(n):
##    axes[i].plot(new[:, i])
##plt.show()
#
#import numpy as np
#import pandas as pd
#np.save('/home/mcolliga/data.npy', pd.DataFrame(new).values)
