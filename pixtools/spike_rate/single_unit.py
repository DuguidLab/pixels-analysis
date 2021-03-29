"""
Plot a histogram of spike times around a specified event.
"""

import seaborn as sns
import matplotlib.pyplot as plt


def single_unit_spike_rate(data, ax=None, cell_id=None, ci=95):
    trials = data.columns.get_level_values('trial').unique()

    if not ax:
        ax = plt.gca()

    val_data = data.stack().reset_index()

    p = sns.lineplot(
        data=val_data,
        x='level_0',
        y=0,
        ci=ci,
        ax=ax,
        linewidth=0.5,
    )
    p.autoscale(enable=True, tight=True)
    #p.set_xticks([])

    palette = sns.color_palette()
    if cell_id is not None:
        p.text(
            0.95, 0.95,
            cell_id,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            color=palette[0],
        )

    #p.get_yaxis().get_label().set_visible(False)
    p.get_xaxis().get_label().set_visible(False)
    p.axvline(c=palette[1], ls='--', linewidth=0.5)
