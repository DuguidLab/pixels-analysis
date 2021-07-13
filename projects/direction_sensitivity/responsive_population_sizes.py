import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

from pixels import Experiment
from pixels.behaviours.pushpull import ActionLabels, Events, PushPull
from pixtools import spike_rate, utils

fig_dir = Path('~/duguidlab/Direction_Sensitivity/neuropixels_figures')

mice = [       
    #"C57_1350950",
    "C57_1350951",
    "C57_1350952",
    #"C57_1350953",  # MI done, needs curation
    "C57_1350954",
    #"C57_1350955",  # corrupted video
]

exp = Experiment(
    mice,
    PushPull,
    '~/duguidlab/Direction_Sensitivity/Data/Neuropixel',
)

rec_num = 0

units = exp.select_units(
    min_depth=550,
    max_depth=1200,
    name="550-1200",
)

for s, session in enumerate(exp):
    # Load coding accuracies for session
    results_npy = session.interim / "cache" / "naive_bayes_results_direction.npy"
    randoms_npy = session.interim / "cache" / "naive_bayes_random_direction.npy"
    assert results_npy.exists() and randoms_npy.exists()

    results = np.load(results_npy)
    randoms = np.load(randoms_npy)

    # Find response period which we are interested in
    action_labels = session.get_action_labels()
    actions = action_labels[rec_num][:, 0]
    events = action_labels[rec_num][:, 1]
    push_starts = np.where(np.bitwise_and(actions, ActionLabels.rewarded_push_good_mi))[0]
    pull_starts = np.where(np.bitwise_and(actions, ActionLabels.rewarded_pull_good_mi))[0]

    push_durations = []
    for push in push_starts:
        mi_onset = np.where(np.bitwise_and(events[push:push + 8000], Events.motion_index_onset))[0]
        assert mi_onset
        push_end = np.where(np.bitwise_and(events[push:push + 8000], Events.front_sensor_closed))[0]
        push_durations.append(push_end[0] - mi_onset[0])

    pull_durations = []
    for pull in pull_starts:
        mi_onset = np.where(np.bitwise_and(events[pull:pull + 8000], Events.motion_index_onset))[0]
        assert mi_onset
        pull_end = np.where(np.bitwise_and(events[pull:pull + 8000], Events.back_sensor_closed))[0]
        pull_durations.append(pull_end[0] - mi_onset[0])

    # We want to look until the longest median duration of pushes or pulls
    # rounded up to neareset 10 ms so binning make sense
    end = int(max(np.median(push_durations), np.median(pull_durations)))
    end = round(end + 5, -1)
    # and from 300 ms before MI onset, during which the brain is active
    # So we can cut out the period using these
    centre = results.shape[0] // 2
    response_period = results[centre - 300 : centre + end, :, :]
    num_bins = response_period.shape[0] // 10

    # We take coding accuracy to be the mean of the results across trials
    response_period = response_period.mean(axis=1)
    # and also mean it into bins of 10 ms
    binned = np.array(
        [response_period[i * 10:i * 10 + 10] for i in range(num_bins)]
    ).mean(axis=1)

    # Get threshold from randomised data
    random_accuracy = randoms.mean(axis=1)
    threshold = random_accuracy.mean() + random_accuracy.std() * 2

    # See if any units go above threshold in any bin
    supra = (binned > threshold).any(axis=0)
