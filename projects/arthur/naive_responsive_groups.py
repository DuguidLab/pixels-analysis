"""
Plot boxplot of ipsi & contra visual stimulation responsive units in naive mice.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from naive_ipsi_contra_cis import *

rec_num = 0
area = ['m2', 'ppc'][rec_num]

counts_no_bias = []  # Indistinguishable responses
counts_ipsi_bias = []  # Greater change in firing rate with ipsi
counts_contra_bias = []  # Greater change in firing rate with contra
counts_bimodal = []  # Both actions are higher but at different bins
counts_opposite = []  # Opposite direction responses
counts_responsive = []
counts_non_responsive = []

for session in range(len(exp)):
    u_ids1 = ipsi_ci['m2'][session].columns.get_level_values('unit').unique()
    u_ids2 = contra_ci['m2'][session].columns.get_level_values('unit').unique()
    assert not any(u_ids1 - u_ids2)

    no_bias = set()
    ipsi_bias = set()
    contra_bias = set()
    bimodal = set()
    opposite = set()
    non_responsive = set()
    resps_set = set()

    # what i want is to get units from m2, and ppc, then test their responsiveness, then count the number of sigs. so, i need to go through each unit in m2,and count, then do the same for ppc
    for unit in units[session][rec_num]:
        ipsi = ipsi_ci[area][session][unit]
        contra = contra_ci[area][session][unit]

        for bin in ipsi:
            ipsi_bin = ipsi[bin]
            if (ipsi.loc[2.5] > 0).any():
                resps_set.add(unit)
                break
            elif (ipsi.loc[97.5] < 0).any():
                resps_set.add(unit)
                break

        for bin in contra:
            contra_bin = contra[bin]
            if (contra.loc[2.5] > 0).any():
                resps_set.add(unit)
                break
            elif (contra.loc[97.5] < 0).any:
                resps_set.add(unit)
                break

        # Positive responses
        if (0 < ipsi.loc[2.5]).any():  # Positive response to ipsi
            if (0 < contra.loc[2.5]).any():  # Responsive to both actions
                if (contra.loc[97.5] < ipsi.loc[2.5]).any():  # Ipsi response greater
                    if (ipsi.loc[97.5] < contra.loc[2.5]).any():  # Both response greater
                        bimodal.add(unit)
                    else:
                        ipsi_bias.add(unit)
                elif (ipsi.loc[97.5] < contra.loc[2.5]).any():  # Contra response greater
                    contra_bias.add(unit)
                else:
                    no_bias.add(unit)
            elif (contra.loc[97.5] < 0).any():  # Ipsi +ve, Contra -ve
                opposite.add(unit)
            else:
                ipsi_bias.add(unit)

        elif (0 < contra.loc[2.5]).any():  # Positive response to contra only
            if (ipsi.loc[97.5] < 0).any():  # Contra +ve, Ipsi -ve
                opposite.add(unit)
            else:
                contra_bias.add(unit)

        else:  # No negative response
            if (ipsi.loc[97.5] < 0).any():  # Ipsi -ve
                if (contra.loc[97.5] < 0).any():  # Contra -ve too
                    if (contra.loc[97.5] < ipsi.loc[2.5]).any():  # Contra response more -ve
                        if (ipsi.loc[97.5] < contra.loc[2.5]).any():  # Both response more -ve
                            bimodal.add(unit)
                        else:
                            contra_bias.add(unit)
                    elif (ipsi.loc[97.5] < contra.loc[2.5]).any():  # Ipsi response more -ve
                        ipsi_bias.add(unit)
                    else:
                        no_bias.add(unit)
                else:
                    ipsi_bias.add(unit)
            elif (contra.loc[97.5] < 0).any():  # Contra -ve
                contra_bias.add(unit)
            else:
                non_responsive.add(unit)
        if unit not in non_responsive:
            resps_set.add(unit)

    num_units = len(units[session][rec_num])
    counts_non_responsive.append(len(non_responsive) / num_units)

#    num_resp = len(resps_set)
    num_resp = sum([len(no_bias), len(ipsi_bias), len(contra_bias), len(bimodal), len(opposite)])
    if num_resp != 0:
#    assert num_resp == \
        counts_no_bias.append(len(no_bias) / num_resp)
        counts_ipsi_bias.append(len(ipsi_bias) / num_resp)
        counts_contra_bias.append(len(contra_bias) / num_resp)
        counts_bimodal.append(len(bimodal) / num_resp)
        counts_opposite.append(len(opposite) / num_resp)


#_, axes = plt.subplots(1, 2, sharey=True)
#
#counts = {
#    "Non": counts_non_responsive,
#    "Move": counts_movement_responsive,
#    "Reward": counts_reward_responsive,
#}
#count_df = pd.DataFrame(counts).melt(value_name="Proportion", var_name="Group")
#
#sns.boxplot(
#    data=count_df,
#    x="Group",
#    y="Proportion",
#    ax=axes[0],
#    linewidth=2.5,
#)
#sns.swarmplot(
#    data=count_df,
#    x="Group",
#    y="Proportion",
#    ax=axes[0],
#    linewidth=2.5,
#    color=".25",
#)

biases = {
    "No bias": counts_no_bias,
    "Ipsi": counts_ipsi_bias,
    "Contra": counts_contra_bias,
    "Both": counts_bimodal,
    "Opposite": counts_opposite,
}
bias_df = pd.DataFrame(biases).melt(value_name="Proportion", var_name="Group")

sns.boxplot(
    data=bias_df,
    x="Group",
    y="Proportion",
#    ax=axes[1],
    linewidth=2.5,
)
sns.swarmplot(
    data=bias_df,
    x="Group",
    y="Proportion",
#    ax=axes[1],
    linewidth=2.5,
    color=".25",
)
plt.ylim([0, 1])
utils.save(fig_dir /f"resp_groups_{area}.pdf")
print(bias_df)
print(resps_set, len(resps_set))
