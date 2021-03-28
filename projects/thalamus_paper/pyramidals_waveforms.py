from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from pixels import Experiment
from pixels.behaviours.leverpush import LeverPush
from pixtools import clusters, utils


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

fig_dir = Path('~/duguidlab/visuomotor_control/figures')
sns.set(font_scale=0.4)
rec_num = 0

waveforms = exp.get_spike_waveforms(
    min_depth=500,
    max_depth=1200,
    min_spike_width=0.4,
)

for session in range(len(exp)):
    clusters.session_waveforms(waveforms[session][rec_num])
    name = exp[session].name
    plt.suptitle(f'Session {name} - pyramidal cell waveforms')
    utils.save(fig_dir / f'pyramidal_cell_waveforms_{name}.png')
