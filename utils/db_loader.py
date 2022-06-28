import numpy as np
import pandas as pd
import sys, os
import pywt
import wfdb
from scipy import signal

target_lv = 4
fs_resampling = 360
duration = 0.15 # 150ms

class DB_loading:
    def __init__(self):
        # path definition
        path_base = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
        self.path_database = path_base + '\\database\\'
        self.report_table = pd.read_excel(path_base + '\\ecg_databases.xlsx')

    ### filtering method
    def dwt_idwt(self, array, wavelet='db3', level=9):
        coeffs = pywt.wavedec(array, wavelet, level=level)
        coeffs[0] = np.zeros_like(coeffs[0])
        array_filtered = pywt.waverec(coeffs, wavelet)
        return array_filtered

    def lowpass_filter(self, array, fs, cutoff=40, order=5, remove_lag=True):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        sos = signal.butter(order, normal_cutoff, btype='low', output='sos')
        array_filtered = signal.sosfilt(sos, array)
        if remove_lag == True:
            # filter time delay -> to reduce zero lag
            array_filtered = signal.sosfilt(sos, array_filtered[::-1])
            return array_filtered[::-1][:len(array)]
        else:
            return array_filtered[:len(array)]

    def filtering(self, array, fs=fs_resampling):
        array_filtered = array.copy()
        array_filtered = self.dwt_idwt(array_filtered)
        array_filtered = self.lowpass_filter(array_filtered, fs)
        return array_filtered

    ### resampling
    def resample_ecg(self, array, fs_pre, fs_post=fs_resampling):
        t_len = len(array)
        new_len = int(fs_post/fs_pre*t_len)
        array_resampled = signal.resample(array, new_len)
        array_resampled = self.filtering(array_resampled)
        return array_resampled

    def resample_label(self, label, fs_pre, fs_post=fs_resampling):
        label_resampled = (label*fs_post/fs_pre).astype('int')
        return label_resampled

    ### transformation
    def normalization(self, array):
        root_squared_mean = np.mean(array**2)**0.5
        array_norm = array/root_squared_mean/16
        return array_norm

    def sw_transform(self, array):
        len_padding = -len(array)%2**target_lv
        padded = np.pad(array, (0, len_padding), 'edge')
        coeff_swt = pywt.swt(padded, 'sym4', level=target_lv, trim_approx=True)
        # [cAn, cDn, ..., cD2, cD1]
        coeff_swt.reverse()
        # [cD1, cD2, ... ,cDn, cAn]

        feature = coeff_swt[3]
        feature = feature[:len(array)]
        feature = self.normalization(feature)

        diff = np.diff(array, append=array[-1])
        diff = self.normalization(diff)

        merged_feature = np.stack([feature, diff], axis=1)
        return merged_feature

    def transform(self, array, use_swt=True):
        if use_swt == False:
            diff = np.diff(array, append=array[-1])
            diff = self.normalization(diff)
            feature = diff.reshape(-1, 1)
            return feature
        else:
            feature = self.sw_transform(array)
        return feature

    def make_target(self, array, label, w_size=None):
        if w_size is None:
            w_size = int(fs_resampling*duration)
        target = np.zeros(array.shape[0])
        n_label = np.array([x for x in label if x < array.shape[0]]).astype('int')
        target[n_label] = 1
        target = np.convolve(target, np.ones(w_size), mode='same')
        target = np.where(target > 1, 1, target) # ?
        return target

    ### loading
    def return_idx(self, db_name):
        list_idx = self.report_table[(self.report_table['Database'] == db_name) & (self.report_table['Select'] == 1)].index
        return list_idx

    def nan_helper(self, array):
        nans, x = np.isnan(array), lambda z: z.nonzero()[0]
        array[nans] = np.interp(x(nans), x(~nans), array[~nans])
        return array

    def load_annotation(self, ann):
        # https://archive.physionet.org/physiobank/annotations.shtml
        beat_labels = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
        in_beat_labels = np.in1d(ann.symbol, beat_labels)
        sorted_anno = ann.sample[in_beat_labels]
        sorted_anno = np.unique(sorted_anno)
        sorted_anno = sorted_anno[sorted_anno >= 0]
        return sorted_anno

    def load_data(self, order, verbose=False):
        self.order = order
        self.table_loc = self.report_table[self.report_table.index == order].index[0]
        database = self.report_table.loc[self.table_loc, 'Database']
        patient = self.report_table.loc[self.table_loc, 'Patient']
        num = self.report_table.loc[self.table_loc, 'Num']
        path_file = self.path_database+database+'/'+str(patient)

        if verbose == True:
            print('Database : {0}, Patient : {1}'.format(database, patient))

        if database != 'TELE':
            # load data
            record = wfdb.rdsamp(path_file)
            ecg = record[0][:, num]
            fs = record[1]['fs']

            # load annotation
            ann = wfdb.rdann(self.path_database+database+'/'+str(patient), 'atr')
            label = self.load_annotation(ann)
            mask = np.array([])

            if database=='QTDB': # In QTDB database, some files had a length of 224999, not 225000.
                if len(ecg)==224999: 
                    ecg = np.pad(ecg, (0, 1), 'edge')
            elif database=='MIT_BIH': # In MIT_BIH database, ventricular flutter areas were removed.
                if patient == '207': 
                    mask_1 = np.arange(14540, 22100)
                    mask_2 = np.arange(87060, 101400)
                    mask_3 = np.arange(554560, 589930)
                    mask = np.concatenate([mask_1, mask_2, mask_3], axis=0)
                    ecg[mask] = np.nan
                    ecg = self.nan_helper(ecg)
                    label = np.array([x for x in label if x not in mask])

            elif database=='MIT_BIH_ST': # In MIT_BIH_ST database, areas without peak annotation were removed.
                if patient == '319': 
                    mask = np.arange(142300, 186300)
                    ecg[mask] = np.nan
                    ecg = self.nan_helper(ecg)

        else:
            record_temp = []
            with open(path_file+'.dat') as file:
                dat = file.read().splitlines()
                for d in dat:
                    row = np.array(d.split(','), dtype='float')
                    record_temp.append(row)
            record_temp = np.stack(record_temp, axis=1)
            ecg = record_temp[0,:]
            label = record_temp[1,:]
            label = np.where(label == 1)[0]
            fs = 500
            mask = record_temp[2,:] + record_temp[3,:]
            mask = np.where(mask > 0)[0]

            if patient == '244_291': # In TELE database, areas without annotation were masked.
                mask_add = np.arange(0, 7000)
                mask = np.concatenate([mask, mask_add], axis=0)
            if patient == '250_300':
                mask_add = np.arange(0, 8900)
                mask = np.concatenate([mask, mask_add], axis=0)

            ecg[mask] = np.nan
            ecg = self.nan_helper(ecg)
            label = np.array([x for x in label if x not in mask])

        return ecg, label, fs, mask

    # pipeline
    def create_set(self, name_database, use_swt=True): 
        list_idx = self.return_idx(name_database)
        self.metadata_patient = self.report_table.loc[list_idx,:]['Patient'].tolist()

        set_dict = dict()
        set_dict['ecg'] = []
        set_dict['label'] = []
        set_dict['feature'] = []
        set_dict['target'] = []
        set_dict['mask_array'] = []

        for n, idx in enumerate(list_idx):
            print('... Processing  {0} / {1}'.format(n+1, len(list_idx)))
            ecg, label, fs, mask = self.load_data(idx)
            ecg = self.resample_ecg(ecg, fs, fs_resampling)
            label = self.resample_label(label, fs, fs_resampling)
            mask = self.resample_label(mask, fs, fs_resampling)

            feature = self.transform(ecg, use_swt=use_swt)
            target = self.make_target(feature, label)
            mask_array = self.make_target(feature, mask, w_size=25)

            feature_diff = np.abs(feature[:,1]) # ignore flat areas (sum of absolute differences for 2 seconds < 0.1mV)
            area_ignore = np.convolve(feature_diff, np.ones(fs_resampling*2), mode='same')
            area_ignore = np.where(area_ignore < 0.1, 1, 0)
            mask_array += area_ignore
            mask_array = np.where(mask_array>0, 1, 0)

            set_dict['ecg'].append(ecg)
            set_dict['label'].append(label)
            set_dict['feature'].append(feature)
            set_dict['target'].append(target)
            set_dict['mask_array'].append(mask_array)

        return set_dict
