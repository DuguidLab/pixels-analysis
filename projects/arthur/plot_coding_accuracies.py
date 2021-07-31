# Plot coding accuracies for each unit

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

import pandas as pd
import numpy as np

from pixels import Experiment
from pixels.behaviours.reach import VisualOnly, ActionLabels, Events
from pixtools import spike_rate, utils

mice = [       
    'HFR20',
    'HFR22',
    'HFR23',
]

exp = Experiment(
    mice,
    VisualOnly,
    '~/duguidlab/visuomotor_control/neuropixels',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

fig_dir = Path('~/duguidlab/visuomotor_control/AZ_notes/npx-plots/naive')

# Envelope for plots, 95% confidence interval
ci = 95

## Select units
duration = 2
units = exp.select_units(
        min_depth=0, max_depth=1200,
        #min_spike_width=0.4,
        name="cortex0-1200"
        )

# get spike rate for left & right visual stim.
stim_left = exp.align_trials(
    ActionLabels.naive_left,
    Events.led_on,
    'spike_rate',
    units=units,
    duration=duration,
)
stim_right = exp.align_trials(
    ActionLabels.naive_right,
    Events.led_on,
    'spike_rate',
    units=units,
    duration=duration,
)

rec_num = 0
sns.set_context("paper", font_scale=0.5)
sns.set_style("white")
sns.despine()

palette = sns.color_palette()
percs = []

for s, session in enumerate(exp):
    # Load coding accuracies for session
    results_npy = session.interim / "cache" / "naive_bayes_results_direction.npy"
    assert results_npy.exists()
    randoms_npy = session.interim / "cache" / "naive_bayes_random_direction.npy"
    assert randoms_npy.exists()
    results = np.load(results_npy)
    randoms = np.load(randoms_npy)

    # Turn NaNs into 0.5
    results[np.isnan(results)] = 0.5
    accuracies = np.nanmean(results, axis=1)
    randoms[np.isnan(randoms)] = 0.5
    thresholds = np.nanmean(randoms, axis=2) + utils.confidence_interval(randoms, axis=2)

    units = stim_left[s][rec_num].columns.get_level_values("unit").unique()
    num_trials = len(stim_left[s][rec_num][units[0]].columns)
    subplots = utils.Subplots2D(units)

    sigs = []

    for i, unit in enumerate(units):
        ax = subplots.axes_flat[i]

        val_data = stim_left[s][rec_num][unit].stack().reset_index()
        val_data['y'] = val_data[0]
        sns.lineplot(
            data=val_data,
            x='time',
            y='y',
            ci=ci,
            ax=ax,
            linewidth=0.5,
        )

        val_data = stim_right[s][rec_num][unit].stack().reset_index()
        val_data['y'] = val_data[0]
        sns.lineplot(
            data=val_data,
            x='time',
            y='y',
            ci=ci,
            ax=ax,
            linewidth=0.5,
        )

        ax.axvline(c=palette[2], ls='--', linewidth=0.5)

        unit_acc = accuracies[:, i]
        assert 0 <= unit_acc.min() and unit_acc.max() <= 1
        unit_thresh = thresholds[:, i]
        assert 0 <= unit_thresh.min() and unit_thresh.max() <= 1
        ax2 = ax.twinx()
        ax2.plot(np.arange(-1.9, 2.1, 0.1), unit_acc, linewidth=0.5, color="black")
        ax2.plot(np.arange(-1.9, 2.1, 0.1), unit_thresh, linewidth=0.3, color="red")
        ax2.set_ylim([0, 1])

        # Look in -300ms to +1000ms response window
        if np.any(unit_acc[17:30] > unit_thresh[17:30]):
            sigs.append(unit)

        ax.autoscale(enable=True, tight=True)
        ax.get_xaxis().get_label().set_visible(False)
        ax.get_yaxis().get_label().set_visible(False)
        ax2.get_xaxis().get_label().set_visible(False)
        ax2.get_yaxis().get_label().set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.2)
            spine.set_color('gray')
        for spine in ax2.spines.values():
            spine.set_linewidth(0.2)
            spine.set_color('gray')
        ax.tick_params(left=False, labelleft=True, labelbottom=False)
        ax2.tick_params(right=False, labelright=False, labelbottom=False)

    r = len(sigs)
    a = len(units)
    p = round(100 * r / a, 1)
    percs.append(p)
    print(f"{r} / {a} units ({p}%)")
    plt.suptitle(f'stim_left&right firing rate + coding accuracy per (aligned to LED onset)')
    utils.save(fig_dir / f"firing_rate+unit_coding_accuracies_{session.name}")

print(percs)
