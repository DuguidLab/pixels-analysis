"""
Plot boxplot of ipsi & contra visual stimulation responsive units in naive mice.
"""
import matplotlib.pyplot as plt
import seaborn as sns
from naive_ipsi_contra_cis import *

from pixels import  ioutils
from pixtools import correlation

results_dir = Path('~/pixels-analysis/projects/arthur/results')

rec_num = 0
area = ['m2', 'ppc'][rec_num]

counts_no_bias = []  # Indistinguishable responses
counts_ipsi_bias = []  # Greater change in firing rate with ipsi
counts_contra_bias = []  # Greater change in firing rate with contra
counts_bimodal = []  # Both actions are higher but at different bins
counts_opposite = []  # Opposite direction responses
counts_responsive = []
proportion_responsive = []
counts_non_responsive = []
counts_total = []

resps_list = []

for session in range(len(exp)):
    u_ids1 = ipsi_ci["m2"][session].columns.get_level_values("unit").unique()
    u_ids2 = contra_ci["m2"][session].columns.get_level_values("unit").unique()
    assert not any(u_ids1 - u_ids2)

    no_bias = set()
    ipsi_bias = set()
    contra_bias = set()
    bimodal = set()
    opposite = set()
    non_responsive = set()
    resps_ipsi_set = set()
    resps_contra_set = set()
    resps_set = set()

    for unit in units[session][rec_num]:
        ipsi = ipsi_ci[area][session][unit]
        contra = contra_ci[area][session][unit]

        # count total number of responsive units, aligning to ipsi & contra
        for bin in ipsi:
            ipsi_bin = ipsi[bin]
            if ipsi_bin[2.5] > 0:
                resps_ipsi_set.add(unit)
                break
            elif ipsi_bin[97.5] < 0:
                resps_ipsi_set.add(unit)
                break

        for bin in contra:
            contra_bin = contra[bin]
            if contra_bin[2.5] > 0:
                resps_contra_set.add(unit)
                break
            elif contra_bin[97.5] < 0:
                resps_contra_set.add(unit)
                break

        #resps_set = resps_ipsi_set | resps_contra_set
        #counts_responsive.append(len(resps_set))

        # grouping responsive units
        # Positive responses
        if (0 < ipsi.loc[2.5]).any():  # Positive response to ipsi
            if (0 < contra.loc[2.5]).any():  # Responsive to both actions
                if (contra.loc[97.5] < ipsi.loc[2.5]).any():  # Ipsi response greater
                    if (
                        ipsi.loc[97.5] < contra.loc[2.5]
                    ).any():  # Both response greater
                        bimodal.add(unit)
                    else:
                        ipsi_bias.add(unit)
                elif (
                    ipsi.loc[97.5] < contra.loc[2.5]
                ).any():  # Contra response greater
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
                    if (
                        contra.loc[97.5] < ipsi.loc[2.5]
                    ).any():  # Contra response more -ve
                        if (
                            ipsi.loc[97.5] < contra.loc[2.5]
                        ).any():  # Both response more -ve
                            bimodal.add(unit)
                        else:
                            contra_bias.add(unit)
                    elif (
                        ipsi.loc[97.5] < contra.loc[2.5]
                    ).any():  # Ipsi response more -ve
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

     #   if unit not in non_responsive:
     #       resps_set.add(unit)

    resps_list.append(resps_set)
    num_units = len(units[session][rec_num])
    counts_non_responsive.append(len(non_responsive) / num_units)
    counts_total.append(num_units)

    num_resp = len(resps_ipsi_set | resps_contra_set)
    counts_responsive.append(num_resp)
    proportion_responsive.append(num_resp / num_units)
    assert num_resp == sum(
        [len(no_bias), len(ipsi_bias), len(contra_bias), len(bimodal), len(opposite)]
    )
    if num_resp != 0:
        counts_no_bias.append(len(no_bias) / num_resp)
        counts_ipsi_bias.append(len(ipsi_bias) / num_resp)
        counts_contra_bias.append(len(contra_bias) / num_resp)
        counts_bimodal.append(len(bimodal) / num_resp)
        counts_opposite.append(len(opposite) / num_resp)

    print(exp[session].name, area)
    print("total number of units: ", num_units)
    print("total ipsi responsive: ", resps_ipsi_set)
    print("total contra responsive: ", resps_contra_set)
    print("all responsive: ", resps_ipsi_set | resps_contra_set | resps_set)
    print("ipsi bias: ", ipsi_bias)
    print("contra bias: ", contra_bias)
    print("bimodal: ", bimodal)
    print("opposite: ", opposite)
    print("all responsive double-check: ", resps_set)
    print("total number of responsive: ", num_resp)
    print("proportion of responsive: ",  proportion_responsive)

print(resps_list)
resps_df = pd.DataFrame(resps_list).T
ioutils.write_hdf5(results_dir / f'naive_{area}_resps_units.h5', resps_df)

assert False
num_units_df = pd.DataFrame([counts_total, counts_responsive, proportion_responsive], index=['total units', 'responsive units', 'responsive proportion']).T
ioutils.write_hdf5(results_dir / f'naive_{area}_resps_units_count.h5', num_units_df)

bias_df = pd.DataFrame([counts_no_bias, counts_ipsi_bias, counts_contra_bias, counts_opposite], index=['no bias', 'ipsi bias', 'contra bias', 'opposite']).T
ioutils.write_hdf5(results_dir / f'naive_{area}_resps_units_bias_groups.h5', bias_df)

#sns.boxplot(
#    data=num_units_df,
    #y needs to be defined
#)
#sns.swarmplot(
#    data=num_units_df,
#)

sns.boxplot(
    data=bias_df,
    x="Group",
    y="Proportion",
    linewidth=2.5,
)
sns.swarmplot(
    data=bias_df,
    x="Group",
    y="Proportion",
    linewidth=2.5,
    color=".25",
    jitter=0,
)
plt.ylim([0, 1])
utils.save(fig_dir / f"resp_groups_{area}_rolling_bins.pdf")
print(bias_df)
