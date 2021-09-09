import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib_venn import venn2

from pixels.behaviours.pushpull import ActionLabels, Events
from pixtools import utils

from setup import fig_dir, exp, pyramidals, interneurons, rec_num

duration = 4

cell_type_names = ["Pyramidal", "Interneuron"]
units = [pyramidals, interneurons]
pushes = [[], []]
pulls = [[], []]
rewards = [[], []]

for s, ses in enumerate(exp):
    action_labels = ses.get_action_labels()
    assert len(action_labels) == 1
    actions = action_labels[rec_num][:, 0]
    events = action_labels[rec_num][:, 1]

    starts = np.where(np.bitwise_and(actions, ActionLabels.rewarded_push))[0]
    mi = np.where(np.bitwise_and(events, Events.motion_index_onset))[0]
    end = np.where(np.bitwise_and(events, Events.front_sensor_closed))[0]
    pds = []
    for t in starts:
        m = mi[np.where(mi - t >= 0)[0][0]]
        e = end[np.where(end - m >= 0)[0][0]] - m
        pds.append(e)
    push_duration = round(np.median(pds)) / 1000

    starts = np.where(np.bitwise_and(actions, ActionLabels.rewarded_pull))[0]
    mi = np.where(np.bitwise_and(events, Events.motion_index_onset))[0]
    end = np.where(np.bitwise_and(events, Events.back_sensor_closed))[0]
    pds = []
    for t in starts:
        m = mi[np.where(mi - t >= 0)[0][0]]
        e = end[np.where(end - m >= 0)[0][0]] - m
        pds.append(e)
    pull_duration = round(np.median(pds)) / 1000

    # The end of the response window is the median movement duration.
    # The step is arbitrarily chosen.
    # The start of the window is 1 or more steps before the end such that the
    # movement index onset is within the first bin.
    # We want to use the movement completion time of the longest movement
    step = 0.250

    end = max(push_duration, pull_duration)
    start = end
    while start > 0:
        start -= step

    for u, cell_type in enumerate(units):
        push_cis = ses.get_aligned_spike_rate_CI(
            ActionLabels.rewarded_push_good_mi,
            Events.motion_index_onset,
            start=start,
            step=step,
            end=end,
            bl_event=Events.tone_onset,
            bl_start=-1.000,
            units=cell_type[s],
        )
        pushes[u].append(push_cis)

        pull_cis = ses.get_aligned_spike_rate_CI(
            ActionLabels.rewarded_pull_good_mi,
            Events.motion_index_onset,
            start=start,
            step=step,
            end=end,
            bl_event=Events.tone_onset,
            bl_start=-1.000,
            units=cell_type[s],
        )
        pulls[u].append(pull_cis)

        reward_cis = ses.get_aligned_spike_rate_CI(
            ActionLabels.rewarded_pull_good_mi | ActionLabels.rewarded_push_good_mi,
            Events.motion_index_onset,
            start=end,
            step=step,
            end=end + 1.000,
            bl_event=Events.tone_onset,
            bl_start=-1.000,
            units=cell_type[s],
        )
        rewards[u].append(reward_cis)

pushes_pc = pd.concat(pushes[0], axis=1, keys=range(len(exp)))
pulls_pc = pd.concat(pulls[0], axis=1, keys=range(len(exp)))
rewards_pc = pd.concat(rewards[0], axis=1, keys=range(len(exp)))

# First figure
counts_non_responsive = []
counts_movement_responsive = []
counts_reward_responsive = []

# Second figure
counts_no_bias = []  # Indistinguishable responses
counts_push_bias = []  # Greater change in firing rate with push
counts_pull_bias = []  # Greater change in firing rate with pull
counts_opposite = []  # Opposite direction responses

for session in range(len(exp)):
    u_ids1 = pushes_pc[session][rec_num].columns.get_level_values('unit').unique()
    u_ids2 = pulls_pc[session][rec_num].columns.get_level_values('unit').unique()
    assert not any(u_ids1 - u_ids2)

    non_responsive = set()
    movement_responsive = set()
    reward_responsive = set()
    no_bias = set()
    push_bias = set()
    pull_bias = set()
    opposite = set()

    # We actually never use the interneurons here, but with caching it's negligible
    # overhead
    for unit in units[0][session][rec_num]:
        resp = False
        push = pushes_pc[session][rec_num][unit]
        pull = pulls_pc[session][rec_num][unit]

        # Positive responses
        if (0 < push.loc[2.5]).any():  # Positive response to push
            if (0 < pull.loc[2.5]).any():  # Responsive to both actions
                if (pull.loc[97.5] < push.loc[2.5]).any():  # Push response greater
                    if (push.loc[97.5] < pull.loc[2.5]).any():  # Both response greater
                        pull_bin = np.where(push.loc[97.5] < pull.loc[2.5])[0][0]
                        push_bin = np.where(pull.loc[97.5] < push.loc[2.5])[0][0]
                        if pull_bin < push_bin:
                            pull_bias.add(unit)
                        elif pull_bin > push_bin:
                            push_bias.add(unit)
                        else:
                            raise Exception
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
                            pull_bin = np.where(pull.loc[97.5] < push.loc[2.5])[0][0]
                            push_bin = np.where(push.loc[97.5] < pull.loc[2.5])[0][0]
                            if pull_bin < push_bin:
                                pull_bias.add(unit)
                            elif pull_bin > push_bin:
                                push_bias.add(unit)
                            else:
                                raise Exception
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

        if unit in non_responsive:
            # move any reward-responsives out of non_responsive into reward_responsive
            reward = rewards_pc[session][rec_num][unit]
            if (0 < reward.loc[2.5]).any() or (reward.loc[97.5] < 0).any():
                reward_responsive.add(unit)
                non_responsive.remove(unit)

        else:
            movement_responsive.add(unit)

    num_units = len(units[0][session][rec_num])
    assert num_units == \
        sum([len(non_responsive), len(movement_responsive), len(reward_responsive)])
    counts_non_responsive.append(len(non_responsive) / num_units)
    counts_movement_responsive.append(len(movement_responsive) / num_units)
    counts_reward_responsive.append(len(reward_responsive) / num_units)

    num_resp = len(movement_responsive)
    assert num_resp == \
        sum([len(no_bias), len(push_bias), len(pull_bias), len(opposite)])
    counts_no_bias.append(len(no_bias) / num_resp)
    counts_push_bias.append(len(push_bias) / num_resp)
    counts_pull_bias.append(len(pull_bias) / num_resp)
    counts_opposite.append(len(opposite) / num_resp)

    out = exp[session].interim / "cache" / "responsive_groups.pickle"
    with out.open('wb') as fd:
        pickle.dump(
            dict(
                non_responsive=non_responsive,
                movement_responsive=movement_responsive,
                reward_responsive=reward_responsive,
                no_bias=no_bias,
                push_bias=push_bias,
                pull_bias=pull_bias,
                opposite=opposite,
            ),
            fd
        )


_, axes = plt.subplots(1, 2, sharey=True)

counts = {
    "Non": counts_non_responsive,
    "Move": counts_movement_responsive,
    "Reward": counts_reward_responsive,
}
count_df = pd.DataFrame(counts).melt(value_name="Proportion", var_name="Group")
stats = count_df.pivot(columns="Group").describe()
stats.loc["IQR"] = stats.loc["75%"] - stats.loc["25%"]
print(stats)

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
    linewidth=0,
    color=".65",
)

biases = {
    "No bias": counts_no_bias,
    "Push": counts_push_bias,
    "Pull": counts_pull_bias,
    #"Opposite": counts_opposite,  # excluded
}
bias_df = pd.DataFrame(biases).melt(value_name="Proportion", var_name="Group")
stats = bias_df.pivot(columns="Group").describe()
stats.loc["IQR"] = stats.loc["75%"] - stats.loc["25%"]
print(stats)

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
    linewidth=0,
    color=".65",
)
axes[0].set_ylim([0, 1])
utils.save(fig_dir / "resp_groups_to_MC_pyramidals.pdf")
