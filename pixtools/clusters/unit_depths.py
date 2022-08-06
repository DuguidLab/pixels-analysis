"""
Get the depths of all units in an experiment.
"""

import pandas as pd


def unit_depths(exp):
    """
    Parameters
    ==========

    exp : pixels.Experiment
        Your experiment.

    """
    info = exp.get_cluster_info()
    depths = []

    for s, session in enumerate(exp):
        session_depths = {}

        for probe, probe_depth in enumerate(session.get_probe_depth()):
            rec_depths = {}
            rec_info = info[s]
            id_key = 'id' if 'id' in rec_info else 'cluster_id'  # Depends on KS version

            for unit in rec_info[id_key]:
                unit_info = rec_info.loc[rec_info[id_key] == unit].iloc[0].to_dict()
                rec_depths[unit] = probe_depth - unit_info["depth"]

            session_depths[probe] = pd.DataFrame(rec_depths, index=["depth"])

        depths.append(pd.concat(session_depths, axis=1, names=["probe", "unit"]))

    return pd.concat(
        depths, axis=1, names=['session', 'probe', 'unit'], keys=range(len(exp)),
    )
