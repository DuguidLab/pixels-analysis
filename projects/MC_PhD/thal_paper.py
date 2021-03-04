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
    #'MCos5',
    #'MCos9',
    #'MCos29',
    'C57_724',
    #'C57_1288723',
    #'C57_1288727',
]

exp = Experiment(
    mice,
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

exp.set_cache(True)
sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/figures').expanduser()

#exp.sort_spikes()
#exp.process_behaviour()
#exp.process_lfp()
#exp.process_spikes()
#exp.extract_spikes()
#exp.process_motion_tracking()

#hits = exp.align_trials(ActionLabels.rewarded_push, Events.back_sensor_open, 'behaviour')

#clusters.depth_profile(exp, curated=False)
#plt.ylim([-250, 4000])
#plt.show()

#spike_times.population_heatmap(signal.from_spike_times(hits))


# Plotting all behavioural data channels for session 1, trial 3
hits = exp.align_trials(
    ActionLabels.rewarded_push,  # This selects which trials we want
    Events.back_sensor_open,  # This selects what event we want them aligned to 
    'spike',  # And this selects what kind of data we want
    duration=20,
    raw=True,
)

plt.figure()
fig, axes = plt.subplots(6, 1, sharex=True)
channels = hits.columns.get_level_values('unit').unique()
trial = 3
session = 0
for i in range(6):
    chan_name = channels[i]
    sns.lineplot(
        data=hits[session][chan_name][trial],
        estimator=None,
        style=None,
        ax=axes[i]
    )
plt.show()

