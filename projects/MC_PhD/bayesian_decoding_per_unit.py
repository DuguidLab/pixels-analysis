from pixels import Experiment
from pixels.behaviours.reach import ActionLabels, Events, VisualOnly
from pixtools.naive_bayes.unit_decoding_accuracy import gen_unit_decoding_accuracies


mice = [       
    'HFR19',
    'HFR20',
    #'HFR21',  # poor quality session
    'HFR22',
    'HFR23',
]

exp = Experiment(
    mice,
    VisualOnly,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

duration = 1

units = exp.select_units(
    min_depth=200,
    max_depth=1200,
    name='200-1200',
)

left = exp.align_trials(
    ActionLabels.naive_left,
    Events.led_on,
    'spike_rate',
    units=units,
    duration=duration,
)

right = exp.align_trials(
    ActionLabels.naive_right,
    Events.led_on,
    'spike_rate',
    units=units,
    duration=duration,
)

# rec_nums
m2 = 0

for s, session in enumerate(exp):
    print(f"Calculating coding accuracies for session {s + 1} / {len(exp)}")

    gen_unit_decoding_accuracies(session, left[s][m2], right[s][m2], "direction")
