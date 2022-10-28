"""
This was used to export some spike times for Julian.

This outputs:

    Dacre et al 2021:
        - unit depths
        - unit median spike widths
        - Spike times for:
            - Hit trials aligned to push onset
            - Laser trials with full pushes aligned to push onset
            - Laser trials with partial pushes aligned to laser onset
            - Laser trials with no pushes aligned to laser onset

    Currie et al 2022:
        - unit depths
        - unit median spike widths
        - Spike times for:
            - Push hit trials aligned to push onset
            - Pull hit trials aligned to push onset
            - Push hit trials aligned to tone onset
            - Pull hit trials aligned to tone onset

"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pixels import Experiment
from pixels.behaviours import leverpush
from pixels.behaviours import pushpull
from pixtools import spike_times, utils
from pixtools.clusters import unit_depths

output = Path("/data/julian_data")

lever_exp = Experiment(
    [
        'C57_724',
        'C57_1288723',
        'C57_1288727',
        'C57_1313404',
    ],
    leverpush.LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

pushpull_exp = Experiment(
    [
        "C57_1350950",
        "C57_1350951",
        "C57_1350952",
        "C57_1350954",
        "C57_1350955",
    ],
    pushpull.PushPull,
    '~/duguidlab/Direction_Sensitivity/Data/Neuropixel',
)

rec_num = 0


## Unit depths

#depths = unit_depths(lever_exp)
#depths.to_csv(output / "dacre2021_unit_depths.csv")
#depths = unit_depths(pushpull_exp)
#depths.to_csv(output / "currie2021_unit_depths.csv")


## Dacre et al 2021 spike times

units = lever_exp.select_units(
    min_depth=500,
    max_depth=1200,
    name="500-1200",
)

#units = lever_exp.select_units(
#    min_depth=100,
#    max_depth=1200,
#    name="100-1200",
#)

select = {
    "units": units,
    "duration": 4,
}

results = {}

results["dacre2021-hits_push_full-back_sensor_open"] = lever_exp.align_trials(
   leverpush.ActionLabels.cued_shutter_push_full,
   leverpush.Events.back_sensor_open,
    'spike_times',
    **select,
)

results["dacre2021-laser_push_full-back_sensor_open"] = lever_exp.align_trials(
    leverpush.ActionLabels.uncued_laser_push_full,
    leverpush.Events.back_sensor_open,
    'spike_times',
    **select,
)

results["dacre2021-laser_push_partial-laser_onset"] = lever_exp.align_trials(
    leverpush.ActionLabels.uncued_laser_push_partial,
    leverpush.Events.laser_onset,
    'spike_times',
    **select,
)

results["dacre2021-laser_nopush-laser_onset"] = lever_exp.align_trials(
    leverpush.ActionLabels.uncued_laser_nopush,
    leverpush.Events.laser_onset,
    'spike_times',
    **select,
)

results["dacre2021-cued_shutter_nopush-tone_onset"] = lever_exp.align_trials(
    leverpush.ActionLabels.cued_shutter_nopush,
    leverpush.Events.tone_onset,
    'spike_times',
    **select,
)

for name, data in results.items():
    data.to_csv(output / f"{name}-spikes.csv")


spike_widths = lever_exp.get_spike_widths(units)
spike_widths.to_csv(output / "dacre2021-spike_widths.csv")


### Currie et al 2022 spike data

units = pushpull_exp.select_units(
    min_depth=100,
    max_depth=1200,
    name="100-1200",
)

select = {
    "units": units,
    "duration": 4,
}

results = {}

results["currie2022-hits_push-back_sensor_open"] = pushpull_exp.align_trials(
    pushpull.ActionLabels.rewarded_push,
    pushpull.Events.back_sensor_open,
    'spike_times',
    **select,
)

results["currie2022-hits_pull-front_sensor_open"] = pushpull_exp.align_trials(
    pushpull.ActionLabels.rewarded_pull,
    pushpull.Events.front_sensor_open,
    'spike_times',
    **select,
)

results["currie2022-hits_push-tone_onset"] = pushpull_exp.align_trials(
    pushpull.ActionLabels.rewarded_push,
    pushpull.Events.tone_onset,
    'spike_times',
    **select,
)

results["currie2022-hits_pull-tone_onset"] = pushpull_exp.align_trials(
    pushpull.ActionLabels.rewarded_pull,
    pushpull.Events.tone_onset,
    'spike_times',
    **select,
)

results["currie2022-missed_push-tone_onset"] = pushpull_exp.align_trials(
    pushpull.ActionLabels.missed_push,
    pushpull.Events.tone_onset,
    'spike_times',
    **select,
)

results["currie2022-missed_pull-tone_onset"] = pushpull_exp.align_trials(
    pushpull.ActionLabels.missed_pull,
    pushpull.Events.tone_onset,
    'spike_times',
    **select,
)


for name, data in results.items():
    data.to_csv(output / f"{name}-spikes.csv")

spike_widths = pushpull_exp.get_spike_widths(units)
spike_widths.to_csv(output / "currie2022-spike_widths.csv")
assert 0




## RTs and PDs

# name: (Action, from event, to event)
#lever_trials = {
#    "dacre2021-hit_push_full": (
#        leverpush.ActionLabels.cued_shutter_push_full,
#        leverpush.Events.back_sensor_open,
#        leverpush.Events.front_sensor_closed,
#    ),
#    "dacre2021-laser_push_full": (
#        leverpush.ActionLabels.uncued_laser_push_full,
#        leverpush.Events.back_sensor_open,
#        leverpush.Events.front_sensor_closed,
#    ),
#}
#
#pushpull_trials = {
#    "currie2022-hits_push": (
#        pushpull.ActionLabels.rewarded_push,
#        pushpull.Events.back_sensor_open,
#        pushpull.Events.front_sensor_closed,
#    ),
#    "currie2022-hits_pull": (
#        pushpull.ActionLabels.rewarded_pull,
#        pushpull.Events.front_sensor_open,
#        pushpull.Events.back_sensor_closed,
#    ),
#}
#
#trials = {
#    lever_exp: lever_trials,
#    pushpull_exp: pushpull_trials,
#}
#
#for exp, trials in trials.items():
#    for trial_name, (label, event_from, event_to) in trials.items():
#        rts = {}
#        pds = {}
#
#        for ses in exp:
#            action_labels = ses.get_action_labels()
#            actions = action_labels[rec_num][:, 0]
#            events = action_labels[rec_num][:, 1]
#            trial_starts = np.where(np.bitwise_and(actions, label))[0]
#            ses_rts = []
#            ses_pds = []
#
#            for start in trial_starts:
#                rt = np.where(np.bitwise_and(events[start:start + 5000], event_from))[0][0]
#                end = np.where(np.bitwise_and(events[start:start + 5000], event_to))[0][0]
#
#                ses_rts.append(rt)
#                ses_pds.append(end - rt)
#
#            rts[ses.name] = pd.DataFrame(ses_rts)
#            pds[ses.name] = pd.DataFrame(ses_pds)
#
#        rts_df = pd.concat(rts, axis=1)
#        pds_df = pd.concat(pds, axis=1)
#        rts_df.to_csv(output / f"{trial_name}-rts.csv")
#        pds_df.to_csv(output / f"{trial_name}-pds.csv")
