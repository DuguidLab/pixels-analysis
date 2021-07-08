# Plot coding accuracies for each unit

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

from pixels import Experiment
from pixels.behaviours.pushpull import ActionLabels, Events, PushPull
from pixtools import spike_rate, utils

fig_dir = Path('~/duguidlab/Direction_Sensitivity/neuropixels_figures')

mice = [       
    #"C57_1350950",
    "C57_1350951",
    "C57_1350952",
    #"C57_1350953",  # MI done, needs curation
    "C57_1350954",
    #"C57_1350955",  # corrupted video
]

exp = Experiment(
    mice,
    PushPull,
    '~/duguidlab/Direction_Sensitivity/Data/Neuropixel',
)

rec_num = 0
duration = 4
sns.set_context("paper", font_scale=0.5)
sns.set_style("white")
sns.despine()

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

ci = "sd"
palette = sns.color_palette()

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
    thresholds = np.nanmean(randoms, axis=1) #+ np.std(randoms, axis=1) * 2

    units = pushes[s][rec_num].columns.get_level_values("unit").unique()
    num_trials = len(pushes[s][rec_num][units[0]].columns)
    subplots = utils.Subplots2D(units)

    sigs = []

    for i, unit in enumerate(units):
        ax = subplots.axes_flat[i]

        val_data = pushes[s][rec_num][unit].stack().reset_index()
        val_data['y'] = val_data[0]
        sns.lineplot(
            data=val_data,
            x='time',
            y='y',
            ci=ci,
            ax=ax,
            linewidth=0.5,
        )

        val_data = pulls[s][rec_num][unit].stack().reset_index()
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
        assert 0 < unit_acc.min() and unit_acc.max() < 1
        unit_thresh = thresholds[:, i]
        assert 0 < unit_thresh.min() and unit_thresh.max() < 1
        ax2 = ax.twinx()
        ax2.plot(val_data["time"].unique(), unit_acc, linewidth=0.5, color="black")
        ax2.plot(val_data["time"].unique(), unit_thresh, linewidth=0.3, color="red")
        ax2.set_ylim([0, 1])

        if np.any(unit_acc > unit_thresh):
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

    print(len(sigs), "out of", len(units))
    plt.suptitle(f'Push + pull firing rate + coding accuracy per (aligned to MI onset)')
    utils.save(fig_dir / f"firing_rate+unit_coding_accuracies_{session.name}")
