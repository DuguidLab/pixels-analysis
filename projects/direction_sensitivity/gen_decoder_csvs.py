# Generate a CSV for every unit for the decoder

import numpy as np
from pathlib import Path

from pixels import Experiment
from pixels.behaviours.pushpull import ActionLabels, Events, PushPull
from pixtools import utils

duration = 4
rec_num = 0
output = Path('~/duguidlab/Direction_Sensitivity/Data/Neuropixel/processed/Decoding/decoder_CSVs').expanduser()

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

# I wanted to really check that the same units list is seen in all variables
all_unit_ids = [u for s in units for r in s for u in r]
ps_units = pushes.columns.get_level_values('unit').unique().values.copy()
pl_units = pulls.columns.get_level_values('unit').unique().values.copy()
ps_units.sort()
pl_units.sort()
assert all(ps_units == np.unique(all_unit_ids))
assert all(pl_units == np.unique(all_unit_ids))

# Generate CSV files
neuron_id = 1  # goes into CSV file name

for i, session in enumerate(exp):
    ses_pushes = pushes[i][rec_num]
    ses_pulls = pulls[i][rec_num]

    for unit in ses_pushes.columns.get_level_values('unit').unique():
        u_pushes = ses_pushes[unit]
        table_pushes = np.concatenate([np.zeros((1, u_pushes.shape[1])), u_pushes], axis=0)
        u_pulls = ses_pulls[unit]
        table_pulls = np.concatenate([np.ones((1, u_pulls.shape[1])), u_pulls], axis=0)
        table = np.concatenate([table_pushes, table_pulls], axis=1)
        np.savetxt(output / f'neuron_{neuron_id}.csv', table, delimiter=',')
        neuron_id += 1
