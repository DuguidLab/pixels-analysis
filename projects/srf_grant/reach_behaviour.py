
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pixtools import utils
from reach import Cohort
from reach.session import Outcomes


mouse_ids = [
    "HFR18",
    "HFR29",
    "HFR30",
]

cohort = Cohort.init_from_files(
    data_dir=Path('~/duguidlab/CuedBehaviourAnalysis/Data/TrainingJSON').expanduser(),
    mouse_ids=mouse_ids,
)


# Find best days
results = pd.DataFrame(cohort.get_results())
results['correct'] = results['correct_l'] + results['correct_r']
results['incorrect'] = results['incorrect_l'] + results['incorrect_r']

best_days = []
max_rewards = []
incorrects = []

for mouse in mouse_ids:
    mouse_results = results[results['mouse_id'] == mouse]
    max_reward = mouse_results['correct'].max()
    best_day = mouse_results[mouse_results['correct'] == max_reward]
    incorrects.append(best_day['incorrect'].values[0])
    max_rewards.append(max_reward)
    best_days.append(best_day['day'].values[0])


# Get trial data from best days
trials = pd.DataFrame(cohort.get_trials())
trials["grasp_latency"] = trials.end - trials.start
latencies = []

for i, mouse in enumerate(mouse_ids):
    mouse_trials = trials[trials['mouse_id'] == mouse]
    day_trials = mouse_trials[mouse_trials['day'] == best_days[i]]
    corrects = day_trials[day_trials['outcome'] == Outcomes.CORRECT]
    latencies.append(corrects["grasp_latency"])

all_latencies = pd.concat(latencies, axis=1, keys=mouse_ids)

_, axes = plt.subplots(2, 2)
sns.histplot(data=all_latencies, ax=axes[0][0])
sns.boxplot(data=all_latencies, ax=axes[0][1])
sns.violinplot(data=all_latencies, ax=axes[1][0])
sns.stripplot(data=all_latencies, ax=axes[1][1])
utils.save("~/duguidlab/visuomotor_control/figures/srf_grant/reach_behaviour_grasp_latency.pdf")
