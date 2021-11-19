import matplotlib.pyplot as plt
import seaborn as sns

from pixtools import utils

from setup import *
#exp.process_behaviour()
exp.process_spikes()
#exp.assess_noise()

# Plotting spike data from session 1, trial 8, units 101 to 110
hits = exp.align_trials(
    ActionLabels.correct_left | ActionLabels.correct_right,
    Events.led_off,
    'spike',
    duration=10,
)

rows = 25
_, axes = plt.subplots(rows, 1, sharex=True)
rec_num = 0
session = 0

for trial in range(10):
    for i in range(rows):
        sns.lineplot(
            data=hits[session][rec_num][101 + i][trial],
            estimator=None,
            style=None,
            ax=axes[i]
        )
    utils.save(fig_dir / f"noisy_data_to_grasp_{trial}")
