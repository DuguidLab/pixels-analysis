from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush
from pixtools import utils

mice = [
    'C57_724',
    'C57_1288723',
    'C57_1288727',
    'C57_1313404',
]

exp = Experiment(
    mice,
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

widths = exp.get_spike_widths(
    min_depth=500,
    max_depth=1200,
)

sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/figures')
fig, axes = plt.subplots(len(exp) + 1, 1, sharex=True)
plt.tight_layout()
counts = []

pool = True
if pool:
    pooled = []

for session in range(len(exp)):
    values =widths[session]['median_ms'].values 
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
utils.save(fig_dir / 'median_spike_widths_good_deep_units')
