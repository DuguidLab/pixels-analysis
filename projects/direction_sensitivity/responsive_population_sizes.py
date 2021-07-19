from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.pushpull import ActionLabels, Events, PushPull
from pixtools import spike_rate, utils

fig_dir = Path('~/duguidlab/Direction_Sensitivity/neuropixels_figures')

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

rec_num = 0
duration = 4

# We don't care so much about interneurons
units = exp.select_units(
    min_depth=550,
    max_depth=900,
    min_spike_width=0.4,
    name="550-900-pyramidals",
)

start = -0.250
step = 0.250
end = 1.500

pushes = exp.get_aligned_spike_rate_CI(
    ActionLabels.rewarded_push_good_mi,
    Events.motion_index_onset,
    start=start,
    step=step,
    end=end,
    bl_event=Events.tone_onset,
    bl_start=-1.000,
    units=units,
)

pulls = exp.get_aligned_spike_rate_CI(
    ActionLabels.rewarded_pull_good_mi,
    Events.motion_index_onset,
    start=start,
    step=step,
    end=end,
    bl_event=Events.tone_onset,
    bl_start=-1.000,
    units=units,
)

# First figure
counts_non_responsive = []
counts_movement_responsive = []
counts_reward_responsive = []

# Second figure
counts_no_bias = []  # Indistinguishable responses
counts_push_bias = []  # Greater change in firing rate with push
counts_pull_bias = []  # Greater change in firing rate with pull
counts_bimodal = []  # Both actions are higher but at different bins
counts_opposite = []  # Opposite direction responses

for session in range(len(exp)):
    u_ids1 = pushes[session][rec_num].columns.get_level_values('unit').unique()
    u_ids2 = pulls[session][rec_num].columns.get_level_values('unit').unique()
    assert not any(u_ids1 - u_ids2)

    non_responsive = set()
    movement_responsive = set()
    reward_responsive = set()
    no_bias = set()
    push_bias = set()
    pull_bias = set()
    bimodal = set()
    opposite = set()

    for unit in units[session][rec_num]:
        resp = False
        push = pushes[session][rec_num][unit]
        pull = pulls[session][rec_num][unit]

        # Positive responses
        if (0 < push.loc[2.5]).any():  # Positive response to push
            if (0 < pull.loc[2.5]).any():  # Responsive to both actions
                if (pull.loc[97.5] < push.loc[2.5]).any():  # Push response greater
                    if (push.loc[97.5] < pull.loc[2.5]).any():  # Both response greater
                        bimodal.add(unit)
                    else:
                        push_bias.add(unit)
                elif (push.loc[97.5] < pull.loc[2.5]).any():  # Pull response greater
                    pull_bias.add(unit)
                else:
                    no_bias.add(unit)
            elif (pull.loc[97.5] < 0).any():  # Push +ve, Pull -ve
                opposite.add(unit)
            else:
                push_bias.add(unit)

        elif (0 < pull.loc[2.5]).any():  # Positive response to pull only
            if (push.loc[97.5] < 0).any():  # Pull +ve, Push -ve
                opposite.add(unit)
            else:
                pull_bias.add(unit)

        else:  # No positive response
            if (push.loc[97.5] < 0).any():  # Push -ve
                if (pull.loc[97.5] < 0).any():  # Pull -ve too
                    if (pull.loc[97.5] < push.loc[2.5]).any():  # Pull response more -ve
                        if (push.loc[97.5] < pull.loc[2.5]).any():  # Both response more -ve
                            bimodal.add(unit)
                        else:
                            pull_bias.add(unit)
                    elif (push.loc[97.5] < pull.loc[2.5]).any():  # Push response more -ve
                        push_bias.add(unit)
                    else:
                        no_bias.add(unit)
                else:
                    push_bias.add(unit)
            elif (pull.loc[97.5] < 0).any():  # Pull -ve
                pull_bias.add(unit)
            else:
                non_responsive.add(unit)

        if unit not in non_responsive:
            movement_responsive.add(unit)

    num_units = len(units[session][rec_num])
    assert num_units == \
        sum([len(non_responsive), len(movement_responsive), len(reward_responsive)])
    counts_non_responsive.append(len(non_responsive) / num_units)
    counts_movement_responsive.append(len(movement_responsive) / num_units)
    counts_reward_responsive.append(len(reward_responsive) / num_units)

    num_resp = len(movement_responsive)
    assert num_resp == \
        sum([len(no_bias), len(push_bias), len(pull_bias), len(bimodal), len(opposite)])
    counts_no_bias.append(len(no_bias) / num_resp)
    counts_push_bias.append(len(push_bias) / num_resp)
    counts_pull_bias.append(len(pull_bias) / num_resp)
    counts_bimodal.append(len(bimodal) / num_resp)
    counts_opposite.append(len(opposite) / num_resp)


_, axes = plt.subplots(1, 2, sharey=True)

counts = {
    "Non": counts_non_responsive,
    "Move": counts_movement_responsive,
    "Reward": counts_reward_responsive,
}
count_df = pd.DataFrame(counts).melt(value_name="Proportion", var_name="Group")

sns.boxplot(
    data=count_df,
    x="Group",
    y="Proportion",
    ax=axes[0],
    linewidth=2.5,
)
sns.swarmplot(
    data=count_df,
    x="Group",
    y="Proportion",
    ax=axes[0],
    linewidth=2.5,
    color=".25",
)

biases = {
    "No bias": counts_no_bias,
    "Push": counts_push_bias,
    "Pull": counts_pull_bias,
    "Both": counts_bimodal,
    "Opposite": counts_opposite,
}
bias_df = pd.DataFrame(biases).melt(value_name="Proportion", var_name="Group")

sns.boxplot(
    data=bias_df,
    x="Group",
    y="Proportion",
    ax=axes[1],
    linewidth=2.5,
)
sns.swarmplot(
    data=bias_df,
    x="Group",
    y="Proportion",
    ax=axes[1],
    linewidth=2.5,
    color=".25",
)
axes[0].set_ylim([0, 1])
utils.save(fig_dir / "resp_groups.pdf")
