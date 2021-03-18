import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush, ActionLabels, Events


mice = [
    'C57_724',
    'C57_1288723',
    'C57_1288727',
]

exp = Experiment(
    mice,
    LeverPush,
    '~/duguidlab/thalamus_paper/Npx_data',
    '~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON',
)

exp.set_cache(True)
sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/figures').expanduser()

def save(name):
    fig.savefig(fig_dir / name, bbox_inches='tight', dpi=300)

widths = exp.get_spike_widths(
    min_depth=500,
    max_depth=1200,
)

fig, axes = plt.subplots(3, 1, sharex=True)

for session in range(len(exp)):
    values = widths[session][0].values()

    # convert to milliseconds from 30kHz sample points
    values /= 30

    axes[session].hist(
        values,
        bins=range(0, 10),
    )
    axes[session].set_title(f"{exp[session].name}")

plt.suptitle('Median spike widths - good deep units')
save('median_spike_widths_good_deep_units')
