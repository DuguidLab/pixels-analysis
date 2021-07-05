import itertools
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import GaussianNB


def gen_unit_decoding_accuracies(session, data1, data2, name):
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

    """
    # Double check units are the same in both
    units1 = data1.columns.get_level_values('unit').unique()
    units2 = data2.columns.get_level_values('unit').unique()
    assert not np.any(units1.values - units2.values), "Units must be the same for both inputs"

    # Skip if it already exists
    output = session.interim / "cache" / f'naive_bayes_results_{name}.npy'
    if output.exists():
        print(f"Results found for session '{session.name}', name '{name}'. Skipping.")
        return

    duration = len(data1.index)
    results = []

    # Y is a vector specifying whether the trials were of one action or the other
    trials1 = data1.columns.get_level_values('trial').unique()
    trials2 = data2.columns.get_level_values('trial').unique()
    Y = np.concatenate([np.zeros(trials1.shape), np.ones(trials2.shape)])

    per_unit1 = [data1[u] for u in units1]
    per_unit2 = [data2[u] for u in units1]

    args = zip(itertools.repeat(Y), per_unit1, per_unit2)

    # Let's run the for loop across multiple processes to save some time
    with Pool() as pool:

        # We are looking at single neurons: can a given neuron's firing rates tell us
        # whether the trial was one or the other?
        results = pool.starmap(_do_gaussian_nb, args)

    # Save to cache
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.concatenate(results, axis=2))

    # Calculate accuracy for random data to use for thresholding accuracies
    x1 = np.random.normal(0, 1, size=per_unit1[0].shape)
    x2 = np.random.normal(0, 1, size=per_unit2[0].shape)
    test_accuracy = _do_gaussian_nb(Y, pd.DataFrame(x1), pd.DataFrame(x2))
    output = session.interim / "cache" / f'naive_bayes_random_{name}.npy'
    np.save(output, test_accuracy)


def _do_gaussian_nb(Y, x1, x2):
    # X is firing rates of both trial types concatenated
    X = np.concatenate([x1, x2], axis=1)

    # This is our classifier
    classifier = GaussianNB(priors=[0.5, 0.5])
    test_accuracy = np.zeros(X.shape)

    for tp in range(len(x1.index)):  # Iterate over all timepoints
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
