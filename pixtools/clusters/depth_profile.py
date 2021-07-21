"""
Plot depth profile for clusters across sessions.
"""

from pixels import PixelsError

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def depth_profile(exp, session=None, curated=True, group=None, in_brain=True):
    """
    Parameters
    ==========

    exp : pixels.Experiment
        The experiment with data you want to plot.

    session : int, optional
        If passed, just plot the depth profile for this session. Otherwise subplot all
        sessions.

    curated : bool, optional
        Whether to use curated groupings (good, mua, noise) or uncurated groups from
        kilosort. Default: True.

    group : str, optional
        Which group to use. One of 'good', 'mua' or 'noise'. Or None (default), which
        plots them all.

    in_brain : bool, optional
        Whether to only use units that are in the brain. This required the depth of the
        probe to be saved into each session's processed/depth.txt files. Default: True.

    """
    info = exp.get_cluster_info()

    # If a session index was passed, just use that one
    if session is not None:
        info = [info[session]]

    # Take kilosort's groupings if curated == False
    if curated:
        hue = "group"
        for i, ses in enumerate(info):
            for rec in ses:
                if rec['group'].isnull().any():
                    PixelsError(f"{exp[i].name}: Not all units have been assigned groups.")
    else:
        hue = "KSLabel"

    # This flattens the info list in case sessions have multiple recordings and saves
    # their names for axis titles
    info_flat = []
    for i, ses in enumerate(info):
        for r, rec in enumerate(ses):
            if in_brain:
                info_flat.append((exp[i].name, r, rec, exp[i].get_probe_depth()[r]))
            else:
                info_flat.append((exp[i].name, r, rec))

    fig, axes = plt.subplots(1, len(info_flat), sharex=True, sharey=True)
    if isinstance(axes, Axes):
        axes = [axes]

    palette = sns.color_palette()
    colours = dict(
        mua=palette[0],
        noise=palette[1],
        good=palette[2],
        unsorted=palette[3],
    )

    for i, rec in enumerate(info_flat):
        if in_brain:
            name, rec_num, data, probe_depth = rec
            data['real_depth'] = probe_depth - data['depth']
            ylabel = "Depth"
            y = "real_depth"
        else:
            name, rec_num, data = rec
            ylabel = "Channel position"
            y = "depth"

        if group:
            data = data.loc[data['group'] == group]

        sns.scatterplot(
            palette=colours,
            x="fr",
            y=y,
            data=data,
            hue=hue,
            ax=axes[i],
            markers=["_", "+"],
        )

        sns.rugplot(
            palette=colours,
            x="fr",
            y=y,
            data=data,
            hue=hue,
            ax=axes[i],
        )

        axes[i].set_xlabel("Firing rate (Hz)")
        axes[i].set_ylabel(ylabel)
        axes[i].set_title(f"{name} rec {rec_num}")

    if in_brain:
        plt.ylim(reversed(plt.ylim()))
        plt.xlim(-10, 50)

    plt.suptitle("Cluster depth profile")

    return fig
