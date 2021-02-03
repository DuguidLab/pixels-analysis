"""
Plot a histogram of spike times around a specified event.
"""

import seaborn as sns
import matplotlib.pyplot as plt

from pixtools.utils import subplots2d


def across_trials_histogram(data, session):
    """
    Plots a histogram for every unit in the given session showing spike.

    data is a multi-dimensional dataframe as returned from Experiment.align_trials.
    """
    units = data[session].columns.get_level_values('unit').unique()

    fig, axes = subplots2d(units, flatten=True, sharex=True, sharey=True)

    for i, unit in enumerate(units):
        sns.histplot(
            data[session][unit],
            binwidth=50,
            ax=axes[i],
            legend=False,
        )
        axes[i].annotate(unit, (360, 1.8))
