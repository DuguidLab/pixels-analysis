# Calculate coding accuracies for each unit and save to cache

from pixels.behaviours.pushpull import ActionLabels, Events, PushPull
from pixtools.naive_bayes.unit_decoding_accuracy import gen_unit_decoding_accuracies

from setup import exp, rec_num, units

duration = 4

pushes = exp.align_trials(
    ActionLabels.rewarded_push_good_mi,
    Events.motion_index_onset,
    'spike_rate',
    duration=duration,
    units=units,
)

pulls = exp.align_trials(
    ActionLabels.rewarded_pull_good_mi,
    Events.motion_index_onset,
    'spike_rate',
    duration=duration,
    units=units,
)

for s, session in enumerate(exp):
    gen_unit_decoding_accuracies(
        session, pushes[s][rec_num], pulls[s][rec_num], "direction", force=True
    )
