from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.pushpull import PushPull
from pixtools import clusters, utils


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

fig_dir = Path('~/duguidlab/Direction_Sensitivity/neuropixels_figures')
sns.set(font_scale=0.4)
rec_num = 0

pyramidals_units = exp.select_units(
    min_depth=550,
    max_depth=900,
    min_spike_width=0.4,
    name="550-900-pyramidals",
)
pyramidals = exp.get_spike_waveforms(units=pyramidals_units)

interneuron_units = exp.select_units(
    min_depth=550,
    max_depth=900,
    max_spike_width=0.35,
    name="550-900-interneurons",
)
interneurons = exp.get_spike_waveforms(units=interneuron_units)

means = True
by_ses = True

# Remove recs with no found units
pyramidals.dropna(axis=1, inplace=True)
interneurons.dropna(axis=1, inplace=True)


if means:
    p_means = pyramidals.mean(level='unit', axis=1)
    i_means = interneurons.mean(level='unit', axis=1)

    # normalise
    p_means = p_means / p_means.abs().max()
    i_means = i_means / i_means.abs().max()
    #p_means = p_means - p_means.min()
    #i_means = i_means - i_means.min()
    #p_means = p_means / p_means.max()
    #i_means = i_means / i_means.max()

    # roll to align troughs
    centre = 1.333333
    istep = p_means.index[1] - p_means.index[0]
    p_roll = ((centre - p_means.loc[1:1.53].idxmin()) / istep).round()
    i_roll = ((centre - i_means.loc[1:1.53].idxmin()) / istep).round()
    for i in p_means:
        p_means[i] = np.roll(p_means[i].values, int(p_roll[i]))
    for i in i_means:
        i_means[i] = np.roll(i_means[i].values, int(i_roll[i]))
    left_clip = int(max(p_roll.max(), i_roll.max()))
    right_clip = int(min(p_roll.min(), i_roll.min()))
    p_means = p_means.iloc[left_clip : right_clip]
    i_means = i_means.iloc[left_clip : right_clip]

    plt.plot(p_means, color='green', alpha=0.5, lw=1)
    plt.plot(i_means, color='orange', alpha=0.5, lw=1)
    utils.save(fig_dir / 'cell_type_waveforms')

if by_ses:
    for session in range(len(exp)):
        name = exp[session].name

        if pyramidals_units[session][0]:
            clusters.session_waveforms(pyramidals[session][rec_num])
            plt.suptitle(f'Session {name} - pyramidal cell waveforms')
            utils.save(fig_dir / f'pyramidal_cell_waveforms_{name}')
        else:
            print("No pyramidals for session", name)

        if interneuron_units[session][0]:
            clusters.session_waveforms(interneurons[session][rec_num])
            plt.suptitle(f'Session {name} - interneuron cell waveforms')
            utils.save(fig_dir / f'interneuron_cell_waveforms_{name}')
        else:
            print("No interneurons for session", name)
