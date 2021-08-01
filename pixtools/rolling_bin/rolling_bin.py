import pandas as pd
import numpy as np

from pixels import Experiment
from pixels.behaviour.reach import ActionLabels, Events, VisualOnly, Reach
from pixtools import utils, spike_rates, clusters, rolling_bins

def get_rolling_bins(units, al, ci_s, ci_e, bl_s, bl_e, increment):
    """
    get confidence intervals with rolling bins.

    parameters:
    ===

    units: units of interest with their spike rates, selected from exp.select_units().

    al: actionlabels.

    ci_s: start of confidence interval calculation.

    ci_e: end of confidence interval calculation.

    bl_s: start of ci baseline calculation.

    bl_e: end of ci baseline calculation.

    increment: overlap between bins.
    """
    for u in units:
        even_bins = exp.get_aligned_spike_rate_CI(
            ActionLabels.al,
            Events.led_on,
            start=ci_s,
            end=ci_e,
            bl_start=bl_s,
            bl_end=bl_e,
            units=u,
        )
        odd_bins = exp.get_aligned_spike_rate_CI(
            ActionLabels.al,
            Events.led_on,
            start=ci_s+increment,
            end=ci_e+increment,
            bl_start=bl_s,
            bl_end=bl_e,
            units=u,
        )
        cis_even = rename_bin(df=even_bins, l=3, names=[0, 200, 400, 600, 800]
        cis_odd = rename_bin(df=odd_bins, l=3, names=[100, 300, 500, 700, 900]
        cis = [pd.concat([cis_even, cis_odd], axis=1)]

        return cis
