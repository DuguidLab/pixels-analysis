import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pixtools import utils

from setup import fig_dir, exp, rec_num, units

sns.set(font_scale=0.4)

widths = exp.get_spike_widths(units=units)

fig, axes = plt.subplots(len(exp) + 1, 1, sharex=True)
plt.tight_layout()
counts = []

# If True, the final histogram will have all data pooled
# Else, it will have the same plots from the other axes overlaid
pool = True
if pool:
    pooled = []

for session in range(len(exp)):
    values = widths[session]['median_ms'].values 
    values = values[~np.isnan(values)]
    axes[session].hist(
        values,
        bins=np.arange(0, 1.2, 0.04),
    )
    axes[session].set_title(f"{exp[session].name}")
    axes[session].set_ylabel("Count")
    axes[session].axvline(x=0.4, c='red')
    count = len(values)
    counts.append(count)
    axes[session].text(
        0.15, 0.95,
        f"Total: {count}",
        horizontalalignment='right',
        verticalalignment='top',
        transform=axes[session].transAxes,
        color='0.3',
    )

    if pool:
        pooled.append(values)
    else:
        axes[-1].hist(
            widths[session]['median_ms'].values,
            bins=np.arange(0, 1.1, 0.05),
            alpha=0.6,
        )

if pool:
    axes[-1].hist(
        np.concatenate(pooled),
        bins=np.arange(0, 1.1, 0.05),
        alpha=0.6,
    )

axes[-1].set_title(f"All sessions")
axes[-1].set_ylabel("Count")
axes[-1].axvline(x=0.4, c='red')
axes[-1].set_xlabel("Median spike width (ms)")
axes[-1].text(
    0.95, 0.15,
    f"Total: {sum(counts)}",
    horizontalalignment='right',
    verticalalignment='top',
    transform=axes[-1].transAxes,
    color='0.3',
)

plt.suptitle('Median spike widths - good units in deep layers')
utils.save(fig_dir / 'median_spike_widths_550-900')
