"""
Neuropixels analysis for the Dacre et al motor thalamus paper.
"""

import os

os.environ['KILOSORT3_PATH'] = '/opt/neuropixels/Kilosort'
os.environ['KILOSORT2_5_PATH'] = os.path.expanduser('~/git/Kilosort2.5')

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from pixels import Experiment, signal
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import clusters, spike_times


mice = [
    'C57_724',
    'C57_1288723',
    'C57_1288727',
    'C57_1313404',
]

exp = Experiment(
    mice,
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/figures').expanduser()

#exp.sort_spikes()
exp.process_behaviour()
#exp.generate_spike_rates()
#exp.process_lfp()
#exp.process_spikes()
#exp.extract_videos()
#exp.process_motion_tracking()



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
