from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import utils

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

sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/figures')
rec_num = 0

select = {
    "min_depth": 500,
    "max_depth": 1200,
    "min_spike_width": 0.4,
    "sigma": 200,
}

hits = exp.get_aligned_spike_rate_CI(
    ActionLabels.cued_shutter_push_full,
    Events.back_sensor_open,
    slice(0.000, 0.200),
    bl_event=Events.tone_onset,
    bl_win=slice(-0.300, -0.100),
    **select,
)

fig, axes = plt.subplots(2, len(exp))
plt.tight_layout()
results = {}
# many of these variables are generated just so i can export to a nice plotting function
# that Josh wrote in matlab
all_deltas = []
all_deltas_stim = []
x_errors = []
y_errors = []
weights = []

for session in range(len(exp)):
    units = hits[session][rec_num].columns.values
    pos = []
    neg = []
    dpos = []
    dneg = []
    deltas = []
    pos_stim = []
    neg_stim = []

    # confidence interval bars
    x_low = []
    x_high = []
    y_low = []
    y_high = []

    for unit in units:
        t = hits[session][rec_num][unit]
        resp = False
        if 0 < t[2.5]:
            pos.append(unit)
            resp = True
            if 0 < stim[session][rec_num][unit][2.5]:
                pos_stim.append(unit)
        elif t[97.5] < 0:
            neg.append(unit)
            resp = True
            if stim[session][rec_num][unit][97.5] < 0:
                neg_stim.append(unit)
        if resp:
            delta = t[50.0]
            t_stim = stim[session][rec_num][unit]
            stim_delta = t_stim[50.0]
            deltas.append((delta, stim_delta))
            x_low.append(delta - t[2.5])
            x_high.append(t[97.5] - delta)
            x_errors.append([delta - t[2.5], t[97.5] - delta])
            y_low.append(stim_delta - t_stim[2.5])
            y_high.append(t_stim[97.5] - stim_delta)
            y_errors.append([stim_delta - t_stim[2.5], t_stim[97.5] - stim_delta])
            all_deltas.append(delta)
            all_deltas_stim.append(stim_delta)

    as_df = pd.DataFrame(deltas, columns=['Cued', 'Stim'])
    print(f"session: {exp[session].name}   +ve: {len(pos)} {len(pos_stim)}  -ve: {len(neg)} {len(neg_stim)}")
    #sns.scatterplot(
    #    data=as_df,
    #    x='Cued',
    #    y='Stim',
    #    ax=axes[0][session],
    #)
    axes[0][session].errorbar(
        as_df['Cued'], as_df['Stim'],
        xerr=[x_low, x_high],
        yerr=[y_low, y_high],
        alpha=0.8,
        linewidth=0.8,
        fmt='none',
    )
    axes[1][0].errorbar(
        as_df['Cued'], as_df['Stim'],
        xerr=[x_low, x_high],
        yerr=[y_low, y_high],
        alpha=0.8,
        linewidth=0.8,
        fmt='.',
    )
    axes[0][session].set_xlim(-10, 50)
    axes[0][session].set_ylim(-10, 50)
    axes[0][session].set_aspect('equal')
    results[exp[session].name] = as_df
    num_resp = len(pos) + len(neg)
    weights.extend([1/num_resp] * num_resp)

all_results = pd.concat(results)
all_results.reset_index(level=0, inplace=True)
#sns.scatterplot(
#    data=all_results,
#    x='Cued',
#    y='Stim',
#    hue='level_0',
#    ax=axes[1][0],
#    legend=None,
#)
axes[1][0].set_aspect('equal')
axes[1][0].set_xlim(-10, 50)
axes[1][0].set_ylim(-10, 50)
plt.suptitle('Cued vs stim push median dHz')

for ax in axes[1][1:]:
    ax.set_visible(False)

utils.save(fig_dir / f'cued_vs_stim_push_median_dHz')
