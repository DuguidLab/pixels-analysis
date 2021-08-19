import numpy as np
import pandas as pd


def cc_matrix(
    m2_units, ppc_units, cc_sig, cc_threshold=0.25, s=None, naive=True, pos=True
):
    """
    get correlation coefficient matrix, units that are significantly correlated
    with each other from cortical areas, number of correlated units count from
    all sessions.

    parameters:
    ----
    m2_units/ppc_units: list of units with their spike rates, indexed by
    units[session][rec_num].

    cc_sig: significant correlation coefficient matrix calculated from cortical
    areas.

    cc_threshold: threshold applied to correlation coefficient, only cc >
    threshold & < -threshold will be further analysed. Default is 0.25.

    s: side of the visual stimlation. 0 is left, 1 is right.

    naive: naive mice. Default is True.

    pos: positivity of the significant correlation coefficient. True means
    further analysis is on positive correlations, False is on negtative
    correlations. Default is True.

    return:
    ----
    cc_df: df contains unit ids from both areas, and their correlation
    coefficient.

    m2_df: df contains m2 unit ids and the number of ppc units that they are
    correlated with.

    ppc_df: df contains ppc unit ids and the number of m2 units that they are
    correlated with.
    """

    if naive == True:
        if pos == True:
            idx = np.where((cc_sig[:, :, s] >= cc_threshold))
        else:
            idx = np.where((cc_sig[:, :, s] <= -cc_threshold))

        cc = cc_sig[:, :, s][idx]
    else:
        if pos == True:
            idx = np.where((cc_sig >= cc_threshold))
        else:
            idx = np.where((cc_sig <= -cc_threshold))
        cc = cc_sig[idx]

    m2_ids = [m2_units[a] for a in idx[0]]
    ppc_ids = [ppc_units[b] for b in idx[1]]

    cc_matrix = []
    for i in range(len(idx[0])):
        cc_matrix.append((m2_ids[i], ppc_ids[i], cc[i]))

    cc_df = pd.DataFrame(
        cc_matrix, columns=["M2 Unit ID", "PPC Unit ID", "Correlation Coefficient"]
    ).pivot(index="M2 Unit ID", columns="PPC Unit ID", values="Correlation Coefficient")

    m2_df = pd.DataFrame(
        np.unique(m2_ids, return_counts=True), index=["M2 Unit ID", "Count"]
    ).T
    ppc_df = pd.DataFrame(
        np.unique(ppc_ids, return_counts=True), index=["PPC Unit ID", "Count"]
    ).T

    return cc_df, m2_df, ppc_df


def find_max_cc(cc_df, pos=True):

    """
    finds the pair of m2-ppc units that has the highest correlation coefficient.

    parameter:
    ----

    cc_df: dataframe contains all correlation coefficients of a condition from a session.

    pos: bool, positivity of correlation coefficient.
        True: find the highest positive cc, i.e., max.
        False: find the highest negative cc, i.e., min.

    return:
    ----
    max_cc_pair: list, [m2_id, ppc_id, cc]
    """
    if pos == True:
        max_cc = cc_df.max().max()
    else:
        max_cc = cc_df.min().min()

    cc_where = cc_df.isin([max_cc])
    loc = (cc_where.any(), cc_where.any(axis=1))
    pair = (list(loc[1][loc[1] == True].index), list(loc[0][loc[0] == True].index))

    max_cc_pair = list(np.squeeze(pair))

    max_cc_pair.append(max_cc)

    return max_cc_pair


def df_max_cc(max_cc_pair):

    """
    concatenate max_cc_pairs from all sessions.

    parameter:
    ----

    max_cc_pair: list contains max. correlation coefficient of each session and m2-ppc unit ids.
    """
    max_cc_df = pd.DataFrame(max_cc_pair, columns=["M2 Unit ID", "PPC Unit ID", "cc"])

    return max_cc_df


def concat_unit_count(unit_count, pool=False, ipsi=None):
    '''
    concatenate significantly correlated unit counts from all sessions.

    parameter:
    ----
    unit_count: list contains the number of significantly correlated units from PPC/M2, given units in M2/PPC.

    which_session: bool, default is True.
        True: dimension of session will be retained.
        False: sig unit counts from all sessions will be pooled together.

    ipsi: bool, anatomical laterality of m2-ppc.
        True: ipsilateral m2-ppc
        False: contralateral m2-ppc
        Default is None.

    returns:
    ----
    unit_count_df: df, number of sig. correlated units, given units in M2, from all sessions.
    '''
    if not pool:
        if ipsi:
            unit_count_df = pd.concat(unit_count[::2], ignore_index=True)
        else:
            unit_count_df = pd.concat(unit_count[1::2], ignore_index=True)

    else:
        unit_count_df = pd.concat(unit_count, ignore_index=True)

    return unit_count_df


def max_cc_ids(max_cc, pos=True):
    '''
    get the m2-ppc unit pair ids which has max or min correlation coefficient.

    parameters:
    ----
    max_cc: df, contains max_cc vaules and corresponding M2 & PPC ids from all session. saved in results_dir.

    pos: bool, positivity of cc.
        True: find the highest positive cc, i.e., max.
        False: find the highest negative cc, i.e., min.

    returns:
    ----
    m2_id: int

    ppc_id: int
    '''
    if pos == True:
        max_val = max_cc.max(axis=1)[2]
        print('\nThe highest positive cc is', max_val)
    else:
        max_val = max_cc.min(axis=1)[2]  
        print('\nThe highest negative cc is', max_val)

    loc = (max_cc == max_val).any()
    single_max_cc = max_cc[loc[loc == True].index]

    session_idx = single_max_cc.columns[0][1]
    m2_id = single_max_cc.iloc[0][0]
    ppc_id = single_max_cc.iloc[1][0]

    print(f'\nThese m2 unit {m2_id} and ppc unit {ppc_id} are from', single_max_cc.columns[0][0], ', session', session_idx)

    return session_idx, m2_id, ppc_id
