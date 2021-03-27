import random

import matplotlib.pyplot as plt
import seaborn as sns

from pixtools.utils import Subplots2D


def session_waveforms(data, n=100):
    units = data.columns.get_level_values('unit').unique()

    subplots = Subplots2D(units)
    palette = sns.color_palette()

    for i, unit in enumerate(units):
        u_data = data[unit]
        if u_data.shape[1] > n:
            spikes = random.sample(list(u_data.columns.values), k=n)
            u_data = u_data[spikes]

        ax = subplots.axes_flat[i]
        p = sns.lineplot(
            data=u_data,
            ax=ax,
            legend=False,
            linewidth=0.5,
            alpha=0.1,
        )

        p.autoscale(enable=True, tight=True)
        p.set_yticks([])
        p.set_xticks([])
        p.text(
            0.95, 0.95,
            unit,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            color=palette[0],
        )
        p.get_yaxis().get_label().set_visible(False)
        p.get_xaxis().get_label().set_visible(False)
        p.set(facecolor='white')

    to_label.get_yaxis().get_label().set_visible(True)
    to_label.set_ylabel('unit?')
    to_label.get_xaxis().get_label().set_visible(True)
    to_label.set_xlabel('Time (ms)')
    to_label.set_xticks([data.index[0], data.index[-1]])

    return fig
