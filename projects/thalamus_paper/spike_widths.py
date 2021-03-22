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
    axes[session].hist(
        widths[session]['median_ms'].values,
        bins=np.arange(0, 1.1, 0.05),
    )
    axes[session].set_title(f"{exp[session].name}")
    axes[session].set_ylabel("Count")
    axes[session].axvline(x=0.4, c='red')

axes[session].set_xlabel("Median spike width (ms)")
plt.suptitle('Median spike widths - good units in deep layers')
save('median_spike_widths_good_deep_units')
