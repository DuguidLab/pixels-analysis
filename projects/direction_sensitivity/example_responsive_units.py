import matplotlib.pyplot as plt
import seaborn as sns

from pixels.behaviours.pushpull import ActionLabels, Events
from pixtools import spike_rate, spike_times, utils

from setup import fig_dir, exp, pyramidals, rec_num

ci = 95
duration = 4

plt.tight_layout()
sns.set(font_scale=0.4)
sns.set_theme(style="ticks")
palette = sns.color_palette()

pushes = exp.align_trials(
    ActionLabels.rewarded_push_good_mi,
    Events.motion_index_onset,
    'spike_rate',
    duration=duration,
    units=pyramidals,
)

pulls = exp.align_trials(
    ActionLabels.rewarded_pull_good_mi,
    Events.motion_index_onset,
    'spike_rate',
    duration=duration,
    units=pyramidals,
)

pushes_times = exp.align_trials(
    ActionLabels.rewarded_push_good_mi,
    Events.motion_index_onset,
    'spike_times',
    duration=duration,
    units=pyramidals,
)

pulls_times = exp.align_trials(
    ActionLabels.rewarded_pull_good_mi,
    Events.motion_index_onset,
    'spike_times',
    duration=duration,
    units=pyramidals,
)

examples = [
    ("C57_1350951", 169),
    ("C57_1350951", 170),
    ("C57_1350952", 235),
    ("C57_1350950", 252),
]

fig, axes = plt.subplots(4, 4)
axbase = 0

# convenience in case i want to just generate some units quickly
to_plot = []

for i in range(len(examples)):
    mouse, unit = examples[i]
    for ses, m in enumerate(exp):
        if mouse in m.name:
            break

    ax = axes[axbase][i % 4]

    if to_plot and unit not in to_plot:
        ax.set_axis_off()
        ax = axes[axbase + 1][i % 4]
        ax.set_axis_off()
        if i == 3:
            axbase = 2
        continue

    num_pushes = len(pushes[ses][rec_num].columns.get_level_values('trial').unique().values)
    spike_times.single_unit_raster(pushes_times[ses][rec_num][unit], ax=ax)
    spike_times.single_unit_raster(pulls_times[ses][rec_num][unit], ax=ax, start=num_pushes)
    ax.set_axis_off()
    ax.xaxis.set_label_text("")
    ax.yaxis.set_label_text("")
    ax.set_xlim((-1.5 * 1000, 2.0 * 1000))

    ax = axes[axbase + 1][i % 4]
    spike_rate.single_unit_spike_rate(pushes[ses][rec_num][unit], ax=ax, ci=ci)
    spike_rate.single_unit_spike_rate(pulls[ses][rec_num][unit], ax=ax, ci=ci)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([-2, 0, 2])
    ax.xaxis.set_label_text("")
    ax.yaxis.set_label_text("")
    ax.set_xlim((-1.5, 2.0))

    bottom, top = ax.get_ylim()
    d = top / 5
    bottom -= d
    top += d
    ax.set_ylim(bottom, top)

    if i == 3:
        axbase = 2

utils.save(fig_dir / 'example_responsive_units_raster+spike_rate')
