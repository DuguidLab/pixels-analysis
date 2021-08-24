import pickle
import numpy as np

from pixels import ioutils, signal, PixelsError
from pixels.behaviours.pushpull import ActionLabels, Events

from setup import fig_dir, exp, rec_num, units

#exp.extract_videos(force=True)
#exp.draw_motion_index_rois()
#exp.process_behaviour()

#exp[0].add_motion_index_action_label(
#    ActionLabels.rewarded_push,
#    Events.back_sensor_open,
#    0,
#    Events.motion_index_push_onset,
#)


def process_motion_index(self):
    """
    Extract motion indexes from videos using already drawn ROIs.
    """

    ses_rois = {}

    # First collect all ROIs to catch errors early
    for i, recording in enumerate(self.files):
        if 'camera_data' in recording:
            roi_file = self.processed / f"motion_index_ROIs_{i}.pickle"
            if not roi_file.exists():
                raise PixelsError(self.name + ": ROIs not drawn for motion index.")

            # Also check videos are available
            video = self.interim / recording['camera_data'].with_suffix('.avi')
            if not video.exists():
                raise PixelsError(self.name + ": AVI video not found in interim folder.")

            with roi_file.open('rb') as fd:
                ses_rois[i] = pickle.load(fd)

    # Then do the extraction
    for rec_num, recording in enumerate(self.files):
        if 'camera_data' in recording:

            # Get MIs
            video = self.interim / recording['camera_data'].with_suffix('.avi')
            rec_rois = ses_rois[rec_num]
            # The final ROI is the LED sync signal
            led_key = sorted(rec_rois.keys())[-1]
            led_roi = rec_rois[led_key]
            del rec_rois[led_key]
            rec_mi = signal.motion_index(video.as_posix(), rec_rois, self.sample_rate)

            actions = self.get_action_labels()[rec_num][:, 1]
            tone_onsets = np.where(np.bitwise_and(actions, Events.tone_onset))[0]
            tone_offsets = np.where(np.bitwise_and(actions, Events.tone_offset))[0]
            dest_period = np.zeros(actions.shape)
            for onset in tone_onsets:
                dest_period[onset:] = 1
                try:
                    offset = tone_offsets[np.where(tone_offsets > onset)[0][0]]
                    dest_period[offset:] = 0
                except IndexError:
                    break

            led_sync = signal.extract_led_sync_signal(video.as_posix(), led_roi, self.sample_rate)
            lag_start, match = signal.find_sync_lag(dest_period, led_sync)
            lag_end = len(dest_period) - (lag_start + len(led_sync))
            if match < 95:
                print("    The LED MI did not match the cue signal very well.")
            print(f"    Calculated lag from behaviour of LED MI: {(lag_start, lag_end)}")

            if lag_start > 0:
                head = np.zeros((lag_start, rec_mi.shape[1]))
                rec_mi = np.concatenate([head, rec_mi])
            elif lag_start < 0:
                rec_mi = rec_mi[- lag_start:, :]

            if lag_end > 0:
                tail = np.zeros((lag_end, rec_mi.shape[1]))
                rec_mi = np.concatenate([rec_mi, tail])
            elif lag_end < 0:
                rec_mi = rec_mi[:lag_end, :]
            assert rec_mi.shape[0] == actions.shape[0]

            np.save(self.processed / f'motion_index_{rec_num}.npy', rec_mi)


for self in exp:
    process_motion_index(self)
