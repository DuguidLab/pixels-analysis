import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

from pixels import Experiment, signal
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import clusters, spike_rate


mice = [
    #'MCos5',
    #'MCos9',
    #'MCos29',
    'C57_724',
    'C57_1288723',
    'C57_1288727',
    #'C57_1313404',
    #'1300812',
    #'1300810',
    #'1300811',
]

exp = Experiment(
    mice,
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

sns.set(font_scale=0.4)

def save(name):
    fig.savefig(Path('~/duguidlab/visuomotor_control/figures').expanduser() / name, bbox_inches='tight', dpi=300)


## FIRING RATES

rec_num = 0

select = {
    "min_depth": 500,
    "max_depth": 1200,
    "min_spike_width": 0.4,
}

hits = exp.get_aligned_spike_rate_CI(
    ActionLabels.cued_shutter_push_full,
    Events.back_sensor_open,
    slice(-99, 100),
    #slice(-249, 250),
    bl_event=Events.tone_onset,
    bl_win=slice(-199, 0),
    #bl_win=slice(-499, 0),
    **select,
)

stim = exp.get_aligned_spike_rate_CI(
    ActionLabels.uncued_laser_push_full,
    Events.back_sensor_open,
    #slice(-99, 100),
    slice(-249, 250),
    bl_event=Events.laser_onset,
    #bl_win=slice(-199, 0),
    bl_win=slice(-499, 0),
    **select,
)

fig, axes = plt.subplots(2, len(exp))
results = {}

for session in range(len(exp)):
    units = hits[session][rec_num].columns.values
    pos = []
    neg = []
    dpos = []
    dneg = []
    deltas = []

    for unit in units:
        t = hits[session][rec_num][unit]
        resp = False
        if 0 < t[2.5]:
            pos.append(unit)
            delta = t[50.0]
            resp = True
        elif t[97.5] < 0:
            neg.append(unit)
            delta = t[50.0]
            resp = True
        if resp:
            stim_delta = stim[session][rec_num][unit][50.0]
            deltas.append((delta, stim_delta))

    as_df = pd.DataFrame(deltas, columns=['Cued', 'Stim'])
    sns.scatterplot(
        data=as_df,
        x='Cued',
        y='Stim',
        ax=axes[0][session],
    )
    axes[0][session].set_aspect('equal')
    results[exp[session].name] = as_df
    print(f"session: {exp[session].name}   +ve: {len(pos)}   -ve: {len(neg)}")

all_results = pd.concat(results)
all_results.reset_index(level=0, inplace=True)
sns.scatterplot(
    data=all_results,
    x='Cued',
    y='Stim',
    hue='level_0',
    ax=axes[1][0],
    legend=None,
)
axes[1][0].set_aspect('equal')
plt.suptitle('Cued vs stim push median dHz')

for ax in axes[1][1:]:
    ax.set_visible(False)

save(f'cued_vs_stim_push_median_dHz.png')
