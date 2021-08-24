import numpy as np
import pickle
import shutil
from scipy.io import loadmat

from pixels import ioutils, signal, PixelsError
from pixels.behaviours.pushpull import ActionLabels, Events

from setup import fig_dir, exp, rec_num, units

# These are the values we are inserting
new_actions = [ActionLabels.rewarded_push_good_mi, ActionLabels.rewarded_pull_good_mi]

for session in exp:
    # Load data from matlab
    cueIDmat = loadmat(session.interim / "cueID.mat")  # 1s for pushes, 2s for pulls
    MIonsets_mat = loadmat(session.interim / "MIonsets.mat")  # onsets in milliseconds
    cueID = cueIDmat["cueID"][0]
    MIonsets = MIonsets_mat["allMIonsets"][0]
    assert len(cueID) == len(MIonsets)

    # Back up original action labels file
    output = session.processed / session.files[rec_num]['action_labels']
    backup = output.with_suffix('.backup.npy')
    if backup.exists():
        # Copy backup into place and operate on copy
        shutil.copy(backup, output)
    else:
        # No backup found; main file is unchanged, back it up
        shutil.copy(output, backup)

    # Insert data into action labels
    action_labels = session.get_action_labels()
    scan_duration = session.sample_rate * 8
    half = scan_duration // 2
    actions = action_labels[rec_num][:, 0]
    events = action_labels[rec_num][:, 1]

    if "C57_1350954" in session.name:
        actions[873595:] = 0

    for i, label in enumerate([ActionLabels.rewarded_push, ActionLabels.rewarded_pull]):
        trial_starts = np.where(np.bitwise_and(actions, label))[0]

        onsets = MIonsets[cueID == i + 1]  # 1 for push, 2 for pull
        assert len(onsets) == len(trial_starts)

        for t, start in enumerate(trial_starts):
            onset = onsets[t]
            if onset is not np.nan and onset > 0:
                actions[start] |= new_actions[i]
                events[int(start + onset)] |= Events.motion_index_onset

    np.save(output, action_labels[rec_num])
