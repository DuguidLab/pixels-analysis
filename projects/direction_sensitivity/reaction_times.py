"""
Neuropixels analysis for the direction sensitivity project.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.pushpull import PushPull, ActionLabels, Events

mice = [       
    "C57_1350950",
    "C57_1350951",
    "C57_1350952",
    #"C57_1350953",
    "C57_1350954",
]


exp = Experiment(
    mice,
    PushPull,
    '~/duguidlab/Direction_Sensitivity/Data/Neuropixel',
)
rec_num = 0


all_push_rts = []
med_push_rts = []
all_pull_rts = []
med_pull_rts = []

for i, ses in enumerate(exp):
    action_labels = ses.get_action_labels()

    # Just confirm each session has only one recording
    assert len(action_labels) == 1
    actions = action_labels[rec_num][:, 0]
    events = action_labels[rec_num][:, 1]

    starts = np.where(actions == ActionLabels.rewarded_push)[0]
    pushes = np.where(np.bitwise_and(events, Events.back_sensor_open))[0]
    rts = []
    for t in starts:
        lag = np.where(pushes - t >= 0)[0][0]
        rts.append(pushes[lag] - t)

    all_push_rts.extend([{'session': i, 'RT': rt} for rt in rts])
    med_push_rts.append({'session': i, 'median RT': np.median(rts)})

    starts = np.where(actions == ActionLabels.rewarded_pull)[0]
    pulls = np.where(np.bitwise_and(events, Events.front_sensor_open))[0]
    rts = []
    for t in starts:
        lag = np.where(pulls - t >= 0)[0][0]
        rts.append(pulls[lag] - t)

    all_pull_rts.extend([{'session': i, 'RT': rt} for rt in rts])
    med_pull_rts.append({'session': i, 'median RT': np.median(rts)})

push_df = pd.DataFrame(all_push_rts)
push_med_df = pd.DataFrame(med_push_rts)
pull_df = pd.DataFrame(all_pull_rts)
pull_med_df = pd.DataFrame(med_pull_rts)
assert False
sns.scatterplot(x=df['RT'], y=df['session'])
plt.show()
