"""
This outputs reaction times and push durations for the npx sessions.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import utils

output = Path("~/duguidlab/thalamus_paper/Npx_data/julian_data").expanduser()

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

trials = [
    "cued_shutter_push_full",
    "uncued_laser_push_full",
]

rec_num = 0

for trial_type in trials:
    label = getattr(ActionLabels, trial_type)
    push = Events.back_sensor_open
    complete = Events.front_sensor_closed

    rts = {}
    pds = {}

    for ses in exp:
        action_labels = ses.get_action_labels()
        actions = action_labels[rec_num][:, 0]
        events = action_labels[rec_num][:, 1]
        trial_starts = np.where(np.bitwise_and(actions, label))[0]
        ses_rts = []
        ses_pds = []

        for start in trial_starts:
            rt = np.where(np.bitwise_and(events[start:start + 5000], push))[0][0]
            push_end = np.where(np.bitwise_and(events[start:start + 5000], complete))[0][0]

            ses_rts.append(rt)
            ses_pds.append(push_end - rt)

        rts[ses.name] = pd.DataFrame(ses_rts)
        pds[ses.name] = pd.DataFrame(ses_pds)

    rts_df = pd.concat(rts, axis=1)
    pds_df = pd.concat(pds, axis=1)
    rts_df.to_csv(output / f"{trial_type}_rts.csv")
    pds_df.to_csv(output / f"{trial_type}_pds.csv")
