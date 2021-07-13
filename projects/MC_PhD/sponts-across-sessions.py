#!/usr/bin/env python3
"""
This script plots the number of spontaneous reaches each mouse in a cohort performed
each day. Bursts of reaches in quick succession are grouped together and treated as one.
Repeat reaches within these bursts are reaches that were performed within 100ms of the
previous reach.

Usage:
    __file__ /path/to/data_dir mouse1 mouse2
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from reach import Cohort
from reach.session import Outcomes

BURST_WINDOW = 0.100  # seconds


def main(cohort):
    fig, axes = plt.subplots(len(cohort), 1, sharex=True)

    # get DataFrame containing information for every touch
    touches = [
            {'timing': t['end'], 'location': t['spout'],
             'day': t['day'], 'mouse_id': t['mouse_id'],
             'type': 'correct' if t['outcome'] == Outcomes.CORRECT else 'incorrect'}
        for t in cohort.get_trials()
        if t['outcome'] in (Outcomes.CORRECT, Outcomes.INCORRECT)
    ]
    touches.extend(cohort.get_spontaneous_reaches())
    df = pd.DataFrame(touches)
    df.sort_values(
        ['timing', 'location', 'day', 'mouse_id'], axis=0, ascending=True, inplace=True
    )
    new_touches = df.iloc[1:].timing > (df.iloc[:-1].timing.values + BURST_WINDOW)
    new_touches = np.insert(new_touches.values, 0, True)
    df = df[new_touches]
    df.type[df.type.isna()] = 'spontaneous'

    # plot for each mouse
    for i, mouse in enumerate(cohort):
        df_mouse = df[df.mouse_id == mouse.mouse_id]
        for day in range(len(mouse)):  # normalise timings
            first = df_mouse[df_mouse.day == day + 1].timing.min()
            df_mouse.loc[df_mouse.day == day + 1, 'timing'] -= first
        sns.scatterplot(
            data=df_mouse,
            x="timing",
            y="day",
            hue="type",
            hue_order=['correct', 'incorrect', 'spontaneous'],
            marker="|",
            s=100,
            palette=['#0000ff', '#ff0000', '#bbbbbb'],
            ax=axes[i],
            legend=False if i else 'brief',
        )
        axes[i].invert_yaxis()
    axes[2].set_xlabel('Time from first touch (s)')
    fig.suptitle(
        f'Time within session of every touch for mice: {", ".join(cohort.mouse_ids)}',
        fontsize=12,
    )

    plt.show()


if __name__ == '__main__':
    mouse_ids = []
    for i in sys.argv[2:]:
        mouse_ids.append(i)
    if not mouse_ids:
        raise SystemError(
            f'Usage: {__file__} /path/to/data_dir mouse1 mouse2'
        )

    cohort = Cohort.init_from_files(
        data_dir=sys.argv[1],
        mouse_ids=mouse_ids,
    )

    main(cohort)
