# Plot MI aligned to tone

import matplotlib.pyplot as plt
import seaborn as sns

from pixels.behaviours.pushpull import ActionLabels, Events
from pixtools import utils

from setup import fig_dir, exp, rec_num, units

duration = 8

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
