#!/usr/bin/env python3
"""
This script is used to determine "chance levels" of rewards expected for different rates
of random touches to the spouts during a 30 minute session.
"""

import random
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

# number of times to test each number of touches
ITERATIONS = 100

# numbers of touches to test
MIN_TOUCHES = 0
TOUCHES_STEP = 25
MAX_TOUCHES = 1000

# session epochs in milliseconds
DURATION = 1800000
TIMEOUT = 8000
ITI = 5000
CUE = 10000
TRIAL_DURATION = ITI + CUE

random.seed(DURATION)


def estimate_rewards(touches, spouts):
    timepoints = [random.randint(0, DURATION) for _ in range(touches)]
    timepoints = sorted(timepoints)
    resets = 0
    rewards = 0
    now = 0

    while timepoints:
        touch = timepoints.pop(0)
        wait = touch - now
        incorrect = random.randrange(0, spouts) > 0  # accounts for multiple spouts
        if wait % TRIAL_DURATION < ITI or incorrect:
            resets += 1
            while timepoints:
                now = touch
                touch = timepoints.pop(0)
                wait = touch - now
                if wait > TIMEOUT:
                    now += TIMEOUT
                    break
                resets += 1

        else:
            rewards += 1
            now = touch

    return rewards, resets


def main():
    fig, axes = plt.subplots(2, 2, sharex=True, sharey='row')
    rewards = [{}, {}]
    resets = [{}, {}]

    for spouts in range(2):
        for touches in range(MIN_TOUCHES, MAX_TOUCHES + 1, TOUCHES_STEP):
            rw, rs = zip(*[estimate_rewards(touches, spouts + 1) for i in range(ITERATIONS)])
            rewards[spouts][touches] = rw
            resets[spouts][touches] = rs

    sns.barplot(
        data=pd.DataFrame(rewards[0]),
        ax=axes[0, 0],
        seed=DURATION,
        capsize=.4,
    )
    sns.barplot(
        data=pd.DataFrame(rewards[1]),
        ax=axes[0, 1],
        seed=DURATION,
        capsize=.4,
    )
    axes[0, 0].set_ylabel('No.\nrewards', rotation="horizontal", ha="right")

    sns.barplot(
        data=pd.DataFrame(resets[0]),
        ax=axes[1, 0],
        seed=DURATION,
        capsize=.4,
    )
    sns.barplot(
        data=pd.DataFrame(resets[1]),
        ax=axes[1, 1],
        seed=DURATION,
        capsize=.4,
    )
    axes[1, 0].set_ylabel('No.\nresets', rotation="horizontal", ha="right")
    axes[1, 0].set_xlabel('No. touches')

    axes[0, 0].set_title("1 spout")
    axes[0, 1].set_title("2 spouts")
    fig.suptitle(
        'Chance reward count over a {} minute session (95% CI)'.format(DURATION / 60000),
        fontsize=12,
    )

    # this is the only way i managed to get the x axis labels to play nice!!!!
    for label in axes[1, 0].get_xticklabels():
        label.set_visible(False)
    for label in axes[1, 1].get_xticklabels():
        label.set_visible(False)
    for label in axes[1, 0].get_xticklabels()[::4]:
        label.set_visible(True)
    for label in axes[1, 1].get_xticklabels()[::4]:
        label.set_visible(True)
    plt.show()


if __name__ == '__main__':
    main()
