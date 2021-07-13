#!/usr/bin/env python3
"""
Mice often perform multiple reaches in quick succession, which you may want to consider
a single event. To group these bursts and treat them as one, we need a maximum duration
between consecutive reaches to use in clustering them. This script plots the total
number of events that we get if we cluster reaches for each of a range of durations.

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

MSECS_TO_TEST = range(0, 10000, 100)


def main(cohort):
    _, ax = plt.subplots(1, 1, sharex=True)

    touches = [
            {'timing': t['end'], 'location': t['spout'],
             'day': t['day'], 'mouse_id': t['mouse_id']}
        for t in cohort.get_trials()
        if t['outcome'] in (Outcomes.CORRECT, Outcomes.INCORRECT)
    ]
    touches.extend(cohort.get_spontaneous_reaches())
    df = pd.DataFrame(touches)
    df.sort_values(list(df.columns.values), axis=0, ascending=True, inplace=True)

    all_results = []
    for secs in MSECS_TO_TEST:
        new_touches = df.iloc[1:].timing > (df.iloc[:-1].timing.values + secs / 1000)
        new_touches = np.insert(new_touches.values, 0, True)
        df_new = df[new_touches]
        for mouse in cohort:
            df_mouse = df_new[df_new.mouse_id == mouse.mouse_id]
            for day in range(1, len(mouse) + 1):
                all_results.append({
                    'day': day,
                    'mouse_id': mouse.mouse_id,
                    'touches': len(df_mouse[df_mouse.day == day]),
                    'milliseconds': secs,
                })
    
    results = pd.DataFrame(all_results)

    sns.barplot(
        x='milliseconds',
        y='touches',
        data=results,
        ax=ax,
        seed=len(results),
        capsize=.4,
    )
    ax.set_ylabel('No.\ntouches', rotation="horizontal", ha="right")
    ax.set_xlabel('Millisecond window used to group consecutive events')
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_visible(not i % 5)

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
