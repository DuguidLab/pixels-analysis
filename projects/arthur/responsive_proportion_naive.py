"""
Plot responsive (ipsi & contra visual stim.) neurons in naive mice
"""
from naive_ipsi_contra_spike_rate import *

import matplotlib.pyplot as plt
import seaborn as sns


rec_num = 1
data = []
areas = ["M2", "PPC"][rec_num]
as_proportions = True

for session in range(len(exp)):
    name = exp[session].name

    for i, area in enumerate(areas):
        rec_resps_ipsi = ipsi_m2[session][i]
        units = rec_resps_ipsi.columns.get_level_values('unit').unique()
        count = len(units)
        count_resp_ipsi = 0

        for unit in units:
            cis = rec_resps[unit]
            for bin in cis:
                ci_bin = cis[bin]
                if ci_bin[2.5] > 0:
                    count_resp += 1
                    break
                elif ci_bin[97.5] < 0:
                    count_resp += 1
                    break

        if as_proportions:
            data_ipsi.append((count_resp / count, count, area, name))
        else:
            data_ipsi.append((count_resp, count, area, name))

#plot total number of neurons
df = pd.DataFrame(data, columns=["Number of Responsive Neurons", "Total Number of Neurons", "Brain Area", "Session"])
sns.boxplot(data=df, x="Brain Area", y="Total Number of Neurons")
sns.stripplot(data=df, x="Brain Area", y="Total Number of Neurons", color=".25", jitter=0)
plt.ylim(bottom=0)
utils.save(fig_dir / f'Total_Number_of_Neurons')
print(count)
assert False #TODO

#plot proportion of responsive neurons
if as_proportions:
    name = exp[session].name
    df = pd.DataFrame(data, columns=["Proportion of Responsive Neurons", "Total Number of Neurons", "Brain Area", "Session"])
    sns.boxplot(data=df, x="Brain Area", y="Proportion of Responsive Neurons")
    sns.stripplot(data=df, x="Brain Area", y="Proportion of Responsive Neurons", color=".25", jitter=0)
    utils.save(fig_dir / f'Proportion_of_Responsive_Neurons_{name}')
#plot number of responsive neurons
else:
    df = pd.DataFrame(data, columns=["Number of Responsive Neurons", "Total Number of Neurons", "Brain Area", "Session"])
    name = exp[session].name
    sns.boxplot(data=df, x="Brain Area", y="Number of Responsive Neurons")
    sns.stripplot(data=df, x="Brain Area", y="Number of Responsive Neurons", color=".25", jitter=0)
    utils.save(fig_dir / f'Number_of_Responsive_Neurons_naive_{name}')


