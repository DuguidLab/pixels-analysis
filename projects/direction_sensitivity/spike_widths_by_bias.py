import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pixtools import utils

from setup import fig_dir, exp, rec_num, units

sns.set(font_scale=0.4)

widths = exp.get_spike_widths(units=units)

fig, axes = plt.subplots(len(exp) + 1, 1, sharex=True)
plt.tight_layout()

# If True, the final histogram will have all data pooled
# Else, it will have the same plots from the other axes overlaid
pool = True
if pool:
    pooled_no_bias = []
    pooled_bias = []

for i, session in enumerate(exp):
    cached_groups = session.interim / "cache" / "responsive_groups.pickle"
    with cached_groups.open('rb') as fd:
        groups = pickle.load(fd)

    no_bias = groups["no_bias"]
    bias = groups["push_bias"] | groups["pull_bias"]
    values_no_bias = widths[i][widths[i]["unit"].isin(no_bias)]['median_ms'].values
    values_bias = widths[i][widths[i]["unit"].isin(bias)]['median_ms'].values
    values_no_bias = values_no_bias[~np.isnan(values_no_bias)]
    values_bias = values_bias[~np.isnan(values_bias)]

    axes[i].hist(
        values_no_bias,
        bins=np.arange(0, 1.2, 0.04),
    )
    axes[i].hist(
        values_bias,
        bins=np.arange(0, 1.2, 0.04),
    )
    axes[i].set_title(f"{session.name}")
    axes[i].set_ylabel("Count")
    axes[i].axvline(x=0.4, c='red')

    if pool:
        pooled_no_bias.append(values_no_bias)
        pooled_bias.append(values_bias)
    else:
        # TODO Color these by no bias or bias
        axes[-1].hist(
            values_no_bias,
            bins=np.arange(0, 1.1, 0.05),
            alpha=0.6,
        )
        axes[-1].hist(
            values_no_bias,
            bins=np.arange(0, 1.1, 0.05),
            alpha=0.6,
        )

if pool:
    axes[-1].hist(
        np.concatenate(pooled_no_bias),
        bins=np.arange(0, 1.1, 0.05),
        alpha=0.6,
    )
    axes[-1].hist(
        np.concatenate(pooled_bias),
        bins=np.arange(0, 1.1, 0.05),
        alpha=0.6,
    )

axes[-1].set_title(f"All sessions")
axes[-1].set_ylabel("Count")
axes[-1].axvline(x=0.4, c='red')
axes[-1].set_xlabel("Median spike width (ms)")

plt.suptitle('Median spike widths - responsive units in deep layers bias/no bias')
utils.save(fig_dir / 'median_spike_widths_550-900_by_bias')
