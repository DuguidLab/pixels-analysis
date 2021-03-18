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

sns.set(font_scale=0.4)
fig_dir = Path('~/duguidlab/visuomotor_control/figures').expanduser()

def save(name):
    fig.savefig(fig_dir / name, bbox_inches='tight', dpi=300)

widths = exp.get_spike_widths()
fig, axes = plt.subplots(3, 1, sharex=True)
axes[0].hist(widths[0][0].values())
axes[1].hist(widths[1][0].values())
axes[2].hist(widths[2][0].values())
plt.show()
