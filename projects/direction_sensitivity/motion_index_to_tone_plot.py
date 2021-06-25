# Plot MI aligned to tone

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from pixels import Experiment
from pixels.behaviours.pushpull import ActionLabels, Events, PushPull
from pixtools import utils

duration = 8
rec_num = 0
fig_dir = Path('~/duguidlab/visuomotor_control/figures')

mice = [       
    #"C57_1350950",  # no ROIs drawn
    "C57_1350951",  # MI done
    #"C57_1350952",  # MI done
    #"C57_1350953",  # MI done
    #"C57_1350954",  # MI done
    #"C57_1350955",  # no ROIs drawn
]

exp = Experiment(
    mice,
    PushPull,
    '~/duguidlab/Direction_Sensitivity/Data/Neuropixel',
)

hits = exp.align_trials(
    ActionLabels.rewarded_push,
    Events.back_sensor_open,
    'motion_index',
    duration=duration,
)


for i, session in enumerate(exp):
    fig, axes = plt.subplots(10, 1, sharex=True)

    for t in range(10):
        sns.lineplot(
            data=hits[i][rec_num][0][t],
            estimator=None,
            style=None,
            ax=axes[t]
        )
        axes[t].set_title(f"Trial {t}")

    name = session.name
    utils.save(fig_dir / f'motion_index_aligned_to_tone_{name}_{duration}s')
