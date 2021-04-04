from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush
from pixtools import clusters, utils


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

fig_dir = Path('~/duguidlab/visuomotor_control/figures')
sns.set(font_scale=0.4)
rec_num = 0

pyramidals = exp.get_spike_waveforms(
    min_depth=500,
    max_depth=1200,
    min_spike_width=0.4,
)

interneurons = exp.get_spike_waveforms(
    min_depth=500,
    max_depth=1200,
    max_spike_width=0.35,
)

means = True
by_ses = True

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

        clusters.session_waveforms(pyramidals[session][rec_num])
        plt.suptitle(f'Session {name} - pyramidal cell waveforms')
        utils.save(fig_dir / f'pyramidal_cell_waveforms_{name}')

        clusters.session_waveforms(interneurons[session][rec_num])
        plt.suptitle(f'Session {name} - interneuron cell waveforms')
        utils.save(fig_dir / f'interneuron_cell_waveforms_{name}')
