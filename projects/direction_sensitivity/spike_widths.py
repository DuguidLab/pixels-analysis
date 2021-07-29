from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.pushpull import PushPull
from pixtools import utils

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

fig_dir = Path('~/duguidlab/Direction_Sensitivity/neuropixels_figures')
sns.set(font_scale=0.4)
rec_num = 0

units = exp.select_units(
    min_depth=550,
    max_depth=900,
    name="550-900",
)
widths = exp.get_spike_widths(units=units)

fig, axes = plt.subplots(len(exp) + 1, 1, sharex=True)
plt.tight_layout()
counts = []

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
