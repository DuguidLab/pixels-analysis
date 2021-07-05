# Calculate coding accuracies for each unit

from pixels import Experiment
from pixels.behaviours.pushpull import ActionLabels, Events, PushPull
from pixtools.naive_bayes.unit_decoding_accuracy import gen_unit_decoding_accuracies

mice = [       
    #"C57_1350950",  # no ROIs drawn
    "C57_1350951",  # MI done
    "C57_1350952",  # MI done
    #"C57_1350953",  # MI done
    "C57_1350954",  # MI done
    #"C57_1350955",  # no ROIs drawn
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
    max_depth=1200,
    name="550-1200",
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
    print(f"Calculating coding accuracies for session {s + 1} / {len(exp)}")

    gen_unit_decoding_accuracies(session, pushes[s][rec_num], pulls[s][rec_num], "direction")
