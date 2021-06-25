from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events
from pixtools import spike_rate, spike_times, utils

# Sessions - targets
ses_m1_mth = 0
rec_m1 = 0  # session 0
rec_mth = 1  # session 0

ses_ipn_gpi = 1
rec_ipn = 0  # session 1
rec_gpi = 1  # session 1

# Selected units
examples = [
    # session, rec_num, unit ID
    (ses_m1_mth, rec_mth, 91),
    (ses_m1_mth, rec_mth, 133),
    (ses_m1_mth, rec_mth, 134),
    (ses_ipn_gpi, rec_ipn, 210),
    (ses_ipn_gpi, rec_ipn, 141),
    (ses_ipn_gpi, rec_ipn, 223),
]

mice = [       
    #'C57_1343253',  # has no behaviour
    'C57_1343255',
]

exp = Experiment(
    mice,
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

duration = 4
plt.tight_layout()
sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/figures/srf_grant')
out = fig_dir / 'example_units_raster+spike_rate'

firing_rates = exp.align_trials(
    ActionLabels.rewarded_push,
    Events.back_sensor_open,
    'spike_rate',
    duration=duration,
)

times = exp.align_trials(
    ActionLabels.rewarded_push,
    Events.back_sensor_open,
    'spike_times',
    duration=duration,
)

sns.set_theme(style="ticks")
fig, axes = plt.subplots(4, 3)
axbase = 0

# Convenience in case i want to just generate some units quickly
# Make Falsey to plot all units
to_plot = []

# Plot
for i in range(len(examples)):
    ses, rec_num, unit = examples[i]
    ax = axes[axbase][i % 3]

    if to_plot and unit not in to_plot:
        ax.set_axis_off()
        ax = axes[axbase + 1][i % 3]
        ax.set_axis_off()
        if i == 2:
            axbase = 2
        continue

    spike_times.single_unit_raster(times[ses][rec_num][unit], ax=ax)
    ax.set_axis_off()
    ax.xaxis.set_label_text("")
    ax.yaxis.set_label_text("")

    ax = axes[axbase + 1][i % 3]
    spike_rate.single_unit_spike_rate(firing_rates[ses][rec_num][unit], ax=ax)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([-2, 0, 2])
    ax.xaxis.set_label_text("")
    ax.yaxis.set_label_text("")

    bottom, top = ax.get_ylim()
    d = top / 5
    bottom -= d
    top += d
    ax.set_ylim(bottom, top)

    if i == 2:
        axbase = 2

utils.save(out)
