import pandas as pd
from pixels.behaviours.reach import Events
from pixtools import rolling_bin


def get_rolling_bins(exp, units, al, ci_s, step, ci_e, bl_s, bl_e, increment):
    """
    get confidence intervals with rolling bins for different cell types.

    parameters:
    ===

    exp: Experiment class from pixels.

    units: a list of units of interest with their spike rates, selected from
    exp.select_units(). Currently 'pyramidal' & 'interneurons', but can be any
    categories.

    al: actionlabels.

    ci_s: start of confidence interval calculation.

    step: ci bin size.

    ci_e: end of confidence interval calculation.

    bl_s: start of ci baseline calculation.

    bl_e: end of ci baseline calculation.

    increment: overlap between bins.
    """
    if repr(units).count("cortex") == 0:
        cis = []
        for u in units:
            even_bins = exp.get_aligned_spike_rate_CI(
                al,
                Events.led_on,
                start=ci_s,
                step=step,
                end=ci_e,
                bl_start=bl_s,
                bl_end=bl_e,
                units=u,
            )
            odd_bins = exp.get_aligned_spike_rate_CI(
                al,
                Events.led_on,
                start=ci_s + increment,
                step=step,
                end=ci_e + increment,
                bl_start=bl_s,
                bl_end=bl_e,
                units=u,
            )
            cis_even = rolling_bin.rename_bin(
                df=even_bins, l=3, names=[0, 200, 400, 600, 800]
            )
            cis_odd = rolling_bin.rename_bin(
                df=odd_bins, l=3, names=[100, 300, 500, 700, 900]
            )
            cis.append((cis_even, cis_odd))

        # concat even & odd bins within each cell type category, then concat
        # categories to form the final df
        cis_u0 = pd.concat([cis[0][0], cis[0][1]], axis=1)
        cis_u1 = pd.concat([cis[1][0], cis[1][1]], axis=1)
        cis = pd.concat([cis_u0, cis_u1], axis=1, keys=[0, 1], names=["cell type"])

    else:
        even_bins = exp.get_aligned_spike_rate_CI(
            al,
            Events.led_on,
            start=ci_s,
            step=step,
            end=ci_e,
            bl_start=bl_s,
            bl_end=bl_e,
            units=units,
        )
        odd_bins = exp.get_aligned_spike_rate_CI(
            al,
            Events.led_on,
            start=ci_s + increment,
            step=step,
            end=ci_e + increment,
            bl_start=bl_s,
            bl_end=bl_e,
            units=units,
        )
        cis_even = rolling_bin.rename_bin(
            df=even_bins, l=3, names=[0, 200, 400, 600, 800]
        )
        cis_odd = rolling_bin.rename_bin(
            df=odd_bins, l=3, names=[100, 300, 500, 700, 900]
        )
        cis = pd.concat([cis_even, cis_odd], axis=1)

    return cis
