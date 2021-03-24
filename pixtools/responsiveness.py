
import numpy as np

np.random.seed(123)


def significant_CI(values, ss=20, CI=95, bs=10000):
    """
    Check to see if the bootstrapped confidence intervals of `values` do not overlap
    with zero.

    Parameters
    ----------
    values : list
        Values from which to extract bootstrap samples.

    ss : int, optional
        Sample size used when taking sub-samples.

    CI : int/float, optional
        Confidence interval to compare to 0.

    bs : int, optional

    Returns
    -------
    int : 0 if the CI is not different from zero, 1 if it is larger, and -1 if it is
    smaller.

    """
    samples = np.array([np.random.choice(values, size=ss) for i in range(bs)])
    medians = np.median(samples, axis=1)
    lower = (100 - CI) / 2
    upper = 100 - lower
    interval = np.percentile(medians, [lower, 50, upper])
    print(lower, 50, upper)

    if interval[0] <= 0 <= interval[2]:
        # not significant
        return 0

    # return median
    return interval[1]
