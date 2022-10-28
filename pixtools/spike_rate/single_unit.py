"""
Plot a histogram of spike times around a specified event.
"""

import seaborn as sns
import matplotlib.pyplot as plt


def single_unit_spike_rate(
    data, ax=None, cell_id=None, ci=95, palette=None, *args, **kwargs,
):
    """
    Plot the firing rate for a single unit.

    Parameters
    ----------
    data : pd.DataFrame
        This should be the output from `Experiment.align_trials` indexed into for
        session, rec_num and unit. This therefore only contains firing rates of a single
        unit across all trials.

    ax : matplotlib.pyplot.Axis
        The axis to plot the figure on. If not provided, the current axes is used.

    cell_id : Optional[int]
        The cell ID to include in an annotation on the figure.

    ci: int, None or 'sd'
        What to use to generate the envelope.

    Remaining args and kwargs are passed to sns.lineplot.

    """
    trials = data.columns.get_level_values('trial').unique()

    if not ax:
        ax = plt.gca()

    val_data = data.stack().reset_index().rename({0: "Firing rate"}, axis=1)

    if palette is None and "c" not in kwargs and "color" not in kwargs:
        palette = sns.color_palette()

    p = sns.lineplot(
        data=val_data,
        x='time',
        y="Firing rate",
        ci=ci,
        ax=ax,
        linewidth=0.5,
        palette=palette,
        *args,
        **kwargs,
    )
    p.autoscale(enable=True, tight=True)
    p.set_ylabel('Spike Rate (Hz)')
    #p.set_xticks([])

    if cell_id is not None:
        p.text(
            0.95, 0.95,
            cell_id,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            color=palette[0],
        )

    p.axvline(c='black', ls='--', linewidth=0.5)
