"""
Plot depth profile for clusters across sessions.
"""

from pixels import PixelsError

import seaborn as sns
import matplotlib.pyplot as plt


def depth_profile(exp, session=None, curated=True):
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
            info_flat.append((exp[i].name, r + 1, rec))

    fig, axes = plt.subplots(1, len(info_flat), sharex=True, sharey=True)
    palette = sns.color_palette()
    colours = dict(
        mua=palette[0],
        noise=palette[1],
        good=palette[2],
    )

    for i, rec in enumerate(info_flat):
        name, rec_num, data = rec
        sns.scatterplot(
            palette=colours,
            x="fr",
            y="depth",
            data=data,
            hue=hue,
            ax=axes[i],
        )

        sns.rugplot(
            palette=colours,
            x="fr",
            y="depth",
            data=data,
            hue=hue,
            ax=axes[i],
        )

        axes[i].set_xlabel("Firing rate (Hz)")
        axes[i].set_ylabel("Channel position")
        axes[i].set_title(f"{name} rec {rec_num}")

    plt.suptitle("Cluster depth profile")
