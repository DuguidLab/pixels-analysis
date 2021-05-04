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
    "C57_1319786",
    "C57_1319781",
    "C57_1319784",
    "C57_1319783",
    "C57_1319782",
    "C57_1319785",
]

exp = Experiment(
    mice,
    PushPull,
    '~/duguidlab/Direction_Sensitivity/Data/Neuropixel',
)
rec_num = 0


all_rts = []

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

    all_rts.extend([{'session': i, 'RT': rt} for rt in rts])

df = pd.DataFrame(all_rts)
#sns.scatterplot(x=df['RT'], y=df['session'])
#plt.show()
assert False
