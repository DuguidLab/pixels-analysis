import pandas as pd
import numpy as np

def rename_bin(df,l, names):
    """
    rename bins of the ci dataframe into their starting timestamps, i.e.,
    if bin=100, this bin starts at 100ms (aligning to the start of ci calculation).

    parameters:
    ====

    df: pd dataframe that contains cis.

    l: level of the bin in the dataframe.

    names: list of numbers/strings that will be the bin's new names.

    return
    ====
    df: dataframe with new names
    """
    new_names_tuple = dict(zip(df.columns.levels[l], names))
    df = df.rename(columns=new_names_tuple, level=l)

    return df
