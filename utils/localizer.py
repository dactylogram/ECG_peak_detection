import numpy as np
from scipy import signal

fs_resampling = 360
down_ratio = 2**5
window = 0.075 # 75ms
margin = 0.15 # 150ms
refractory = 0.2 # 200ms

class Localizer:
    def __init__(self, label, pred, mask_array, conv_window=window):
        # feature_shape = (batch_size=1, n_channel, sequence_length)
        self.t_margin = int(fs_resampling * margin)
        self.t_window = int(fs_resampling * conv_window)
        self.t_refractory = int(fs_resampling * refractory)
        self.label = label
        self.pred = pred
        self.mask_array = mask_array

    def find_peak(self):
        c_pred = np.convolve(self.pred, np.ones(self.t_window)/self.t_window, mode='same')
        binary = np.where(c_pred > 0.5, c_pred, 0)
        list_peak, _ = signal.find_peaks(binary, height=0.5, distance=self.t_refractory)
        list_peak = list_peak.astype('int')
        list_peak = list_peak[list_peak < self.mask_array.shape[0]]
        self.list_peak = list_peak[self.mask_array[list_peak] != 1]

    def evaluation(self):
        s = self.t_margin
        list_TP_peak = []
        list_FN_peak = []

        for y in self.label:
            y_range = np.arange(y-s, y+s+1)
            inc = np.in1d(y_range, self.list_peak)
            if sum(inc) > 1:
                peak_candidate = y_range[inc]
                peak_candidate_diff = np.abs(peak_candidate-y)
                peak_selected = peak_candidate[np.argmin(peak_candidate_diff)]
                list_TP_peak.append(peak_selected)
            elif sum(inc) == 1:
                list_TP_peak.append(y_range[inc][0])
            else:
                list_FN_peak.append(y)

        self.list_TP_peak = np.unique(list_TP_peak)
        self.list_FP_peak = np.setdiff1d(self.list_peak, list_TP_peak)
        self.list_FN_peak = np.array(list_FN_peak)

        self.TP = self.list_TP_peak.shape[0]
        self.FP = self.list_FP_peak.shape[0]
        self.FN = self.list_FN_peak.shape[0]
        return

    def run(self):
        self.find_peak()
        self.evaluation()
