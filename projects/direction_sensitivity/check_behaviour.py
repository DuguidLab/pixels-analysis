from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.pushpull import PushPull, ActionLabels, Events
from pixtools import spike_rate, utils


mice = [
    #"C57_1319786",
    "C57_1319781",
    #"C57_1319784",
    #"C57_1319783",
]


exp = Experiment(
    mice,
    PushPull,
    '~/duguidlab/Direction_Sensitivity/Data/Neuropixel',
    #'~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)


sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/figures/DS')
duration = 4
rec_num = 0


hits = exp.align_trials(
    ActionLabels.rewarded_push,
    Events.back_sensor_open,
    'behavioural'
)


_, axes = plt.subplots(6, 1, sharex=True)
channels = hits.columns.get_level_values('unit').unique()
trial = 8
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
