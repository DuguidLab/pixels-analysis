import itertools
import pandas as pd
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from numpy.random import default_rng
from sklearn.naive_bayes import GaussianNB

rng = default_rng()


def gen_unit_decoding_accuracies(session, data1, data2, name, bin_size=100):
    """
    Generate a decoding accuracy for each unit representing that unit's ability to
    distinguish between two actions using its firing rate. This looks at each timepoint
    in the firing rates individually and calculates a decoding accuracy score for each
    timepoint, resulting in a 1D array for each unit that aligns to time. If the
    decoding accuracy in this array goes over a threshold, then the unit can be
    considered to reliably reflect, at that timepoint, whether the current trial is of
    one type or the other. The thresold is calculated by calculating a decoding accuracy
    of random data for the same number of trials.

    Parameters
    ----------

    session : `Behaviour` object or subclass
        The session from your experiment to operate on.

    data1 : `pd.DataFrame`
        The firing rate data corresponding to one action. This must be the dataframe
        returned from `Experiment.align_trials` indexed into for both session and
        rec_num, i.e.  containing the firing rates for units from only one recording.

    data2 : `pd.DataFrame`
        Save as data1 but for the second action.

    name : `str`
        The name to use to refer to this application of the decoder. E.g. if we are
        trying to decode a left vs right response, then this might be "direction". This
        is used for caching.

    name : `bin_size`
        Bin size in ms to bin data before running decoding. Default is 100 ms. The
        duration of the data must be divisible by this.

    """
    # Double check units are the same in both
    units1 = data1.columns.get_level_values('unit').unique()
    units2 = data2.columns.get_level_values('unit').unique()
    assert not np.any(units1.values - units2.values), "Units must be the same for both inputs"

    # Skip if it already exists
    output = session.interim / "cache" / f'naive_bayes_results_{name}.npy'
    if output.exists():
        print(f"Results found for session '{session.name}', name '{name}'. Skipping.")
        #return

    results = []

    # Y is a vector specifying whether the trials were of one action or the other
    trials1 = data1.columns.get_level_values('trial').unique()
    trials2 = data2.columns.get_level_values('trial').unique()
    Y = np.concatenate([np.zeros(trials1.shape), np.ones(trials2.shape)])

    # Bin into bins to reduce noise and speed up computation
    duration = len(data1.index)
    bins = duration // bin_size
    d1 = data1.values.reshape((duration, len(units1), len(trials1)))
    d2 = data2.values.reshape((duration, len(units1), len(trials2)))

    per_unit1 = [
        np.mean(
            np.concatenate(
                [d1[i * bin_size:i * bin_size + bin_size, u, :, None] for i in range(bins)],
                axis=2
            ),
            axis=0,
        )
        for u in range(len(units1))
    ]
    per_unit2 = [
        np.mean(
            np.concatenate(
                [d2[i * bin_size:i * bin_size + bin_size, u, :, None] for i in range(bins)],
                axis=2,
            ),
            axis=0,
        )
        for u in range(len(units1))
    ]

    # Do it once outside of the CPU pool to catch any errors that might pop up
    _do_gaussian_nb(Y, per_unit1[0], per_unit2[0])

    # Let's run the for loop across multiple processes to save some time
    with Pool() as pool:

        # We are looking at single neurons: can a given neuron's firing rates tell us
        # whether the trial was one or the other?
        pool_args = zip(itertools.repeat(Y), per_unit1, per_unit2)
        results = pool.starmap(_do_gaussian_nb, pool_args)

        # Do same again 100 times but after randomising Y
        all_randoms = []
        for i in range(1000):
            rng.shuffle(Y)
            pool_args = zip(itertools.repeat(Y), per_unit1, per_unit2)
            randoms = pool.starmap(_do_gaussian_nb, pool_args)
            arr = np.concatenate(randoms, axis=2)
            all_randoms.append(
                np.mean(arr, axis=1)[..., None]  # Extra dimension for concatenating
            )

    # Save to cache
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.concatenate(results, axis=2))
    output = session.interim / "cache" / f'naive_bayes_random_{name}.npy'
    np.save(output, np.concatenate(all_randoms, axis=2))


def _do_gaussian_nb(Y, x1, x2):
    # X is firing rates of both trial types concatenated
    X = np.concatenate([x1.T, x2.T], axis=1)

    # This is our classifier
    classifier = GaussianNB(priors=[0.5, 0.5])
    test_accuracy = np.zeros(X.shape)

    for tp in range(x1.shape[1]):  # Iterate over all timepoints
        tp_values = X[tp]
        tp_values = tp_values.reshape((tp_values.shape[0], 1))

        for tr in range(len(tp_values)):  # Iterate over each trial
            x_train = np.vstack((tp_values[:tr], tp_values[tr + 1:]))
            y_train = np.hstack((Y[:tr], Y[tr + 1:]))
            prob = classifier.fit(x_train, y_train).predict_proba([tp_values[tr]])
            true_label_idx = np.where(classifier.classes_ == Y[tr])[0][0]
            test_accuracy[tp, tr] = prob[0][true_label_idx]

    # Add third dimension for concatenation
    return test_accuracy[..., None]
