# Calculate coding accuracies for each unit and save to cache

from pixels import Experiment
from pixels.behaviours.pushpull import ActionLabels, Events, PushPull
from pixtools.naive_bayes.unit_decoding_accuracy import gen_unit_decoding_accuracies

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

duration = 4
rec_num = 0

units = exp.select_units(
    min_depth=550,
    max_depth=900,
    name="550-900",
)

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
