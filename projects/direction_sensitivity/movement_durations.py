"""
Push and pull duration distributions
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pixels.behaviours.pushpull import ActionLabels, Events
from pixtools import utils

from setup import fig_dir, exp, rec_num

all_push_pds = []
med_push_pds = []
all_pull_pds = []
med_pull_pds = []

for i, ses in enumerate(exp):
    action_labels = ses.get_action_labels()

    # Just confirm each session has only one recording
    assert len(action_labels) == 1
    actions = action_labels[rec_num][:, 0]
    events = action_labels[rec_num][:, 1]

    starts = np.where(np.bitwise_and(actions, ActionLabels.rewarded_push))[0]
    pushes = np.where(np.bitwise_and(events, Events.front_sensor_closed))[0]
    pds = []
    for t in starts:
        lag = np.where(pushes - t >= 0)[0][0]
        pds.append(pushes[lag] - t)

    all_push_pds.extend([{'session': i, 'PD': pd} for pd in pds])
    med_push_pds.append({'session': i, 'median PD': np.median(pds)})

    starts = np.where(np.bitwise_and(actions, ActionLabels.rewarded_pull))[0]
    pulls = np.where(np.bitwise_and(events, Events.back_sensor_closed))[0]
    pds = []
    for t in starts:
        lag = np.where(pulls - t >= 0)[0][0]
        pds.append(pulls[lag] - t)

    all_pull_pds.extend([{'session': i, 'PD': pd} for pd in pds])
    med_pull_pds.append({'session': i, 'median PD': np.median(pds)})

push_df = pd.DataFrame(all_push_pds)
push_med_df = pd.DataFrame(med_push_pds)
pull_df = pd.DataFrame(all_pull_pds)
pull_med_df = pd.DataFrame(med_pull_pds)

fig, axes = plt.subplots(1, 2, sharey=True)

sns.stripplot(x=push_df['session'], y=push_df['PD'], ax=axes[0])
axes[0].set_title("Push durations")

sns.stripplot(x=pull_df['session'], y=pull_df['PD'], ax=axes[1])
axes[1].set_title("Pull durations")

utils.save(fig_dir / "movement_durations")
