from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import spike_rate, spike_times, utils

# This plots the behavioural channels for a given trial for each session
# Settings
duration = 4
trial = 3

mice = [       
    #'C57_1343253',  # has no behaviour
    'C57_1343255',  
]

exp = Experiment(
    mice,
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)
exp.set_cache(False)

sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/figures')

exp.process_behaviour()

hits = exp.align_trials(
    ActionLabels.rewarded_push,
    Events.back_sensor_open,
    'behavioural',
    duration=duration,
)

fig, axes = plt.subplots(5, 1, sharex=True)
channels = hits.columns.get_level_values('unit').unique()

for i, session in enumerate(exp):
    for c in range(5):
        chan_name = channels[c]
        sns.lineplot(
            data=hits[i][rec_num][chan_name][trial],
            estimator=None,
            style=None,
            ax=axes[c]
        )
        ax.set_title(chan_name)

    name = session.name
    utils.save(fig_dir / f'behaviour_alignment_{name}_trial{trial}_{duration}s')
