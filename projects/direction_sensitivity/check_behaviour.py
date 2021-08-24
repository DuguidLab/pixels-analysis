import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pixels.behaviours.pushpull import ActionLabels, Events
from pixtools import spike_rate, utils

from setup import exp, fig_dir, rec_num

sns.set(font_scale=0.4)
fig_dir = fig_dir / 'DS'
duration = 4


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
