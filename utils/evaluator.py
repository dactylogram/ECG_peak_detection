import pandas as pd
import torch
import os

# import sep_conv
# import db_generator
# import localizer
from utils.sep_conv import *
from utils.db_generator import *
from utils.localizer import *
from utils.db_loader import DB_loading

path_utils = os.path.dirname(os.path.abspath(__file__))
path_base = '\\'.join(path_utils.split("\\")[:-1])

n_channel = 2
atrous_rate = [1,3,6,9]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Evaluator:
    def __init__(self, model_name = 'trained_model.pt'):
        self.db_loading = DB_loading()
        self.model_name = model_name

    def load(self, name_database):
        self.name_database = name_database
        self.set_dict = self.db_loading.create_set(self.name_database)
        self.test_loader = Test_Generator(self.set_dict)
        self.test_loader.list_label = self.set_dict['label']
        self.test_loader.list_mask_array = self.set_dict['mask_array']
        self.model_path = path_base + '\\model\\' + self.model_name

        self.set_dict['pred'] = []
        self.set_dict['pred_TP'] = []
        self.set_dict['pred_FP'] = []
        self.set_dict['pred_FN'] = []

    def statistics(self, TP=None, FP=None, FN=None):
        try:
            sensitivity = TP / (TP+FN)
        except:
            sensitivity = 0
        try:
            ppv = TP / (TP+FP)
        except:
            ppv = 0
        try:
            f1 = 2*sensitivity*ppv/(sensitivity+ppv)
        except:
            f1 = 0
        return sensitivity, ppv, f1

    def find_peaks(self):
        torch.cuda.empty_cache()
        model = Sep_conv_detector(n_channel=n_channel, atrous_rate=atrous_rate).to(device)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()

        with torch.no_grad():
            for i, (feature, target) in enumerate(self.test_loader):
                print('... Predicting  {} / {}'.format(i+1, len(self.test_loader)))
                feature = feature.to(device).float()
                target = target.to(device).float()

                output = model(feature)
                pred = torch.sigmoid(output.to('cpu').squeeze()).detach().numpy()
                label = self.test_loader.list_label[i]
                localizer = Localizer(label, pred, self.test_loader.list_mask_array[i])
                localizer.run()
                self.set_dict['pred'].append(pred)
                self.set_dict['pred_TP'].append(localizer.list_TP_peak)
                self.set_dict['pred_FP'].append(localizer.list_FP_peak)
                self.set_dict['pred_FN'].append(localizer.list_FN_peak)
                del feature, target, output
                torch.cuda.empty_cache()

    def report_summary(self):
        all_TP = sum([len(x) for x in self.set_dict['pred_TP']])
        all_FP = sum([len(x) for x in self.set_dict['pred_FP']])
        all_FN = sum([len(x) for x in self.set_dict['pred_FN']])
        sst, ppv, f1 = self.statistics(all_TP, all_FP, all_FN)
        return all_TP, all_FP, all_FN, sst, ppv, f1

    def report_all(self):
        list_TP = [len(x) for x in self.set_dict['pred_TP']]
        list_FP = [len(x) for x in self.set_dict['pred_FP']]
        list_FN = [len(x) for x in self.set_dict['pred_FN']]
        list_sst = []
        list_ppv = []
        list_f1 = []

        for (TP, FP, FN) in zip(list_TP, list_FP, list_FN):
            sst, ppv, f1 = self.statistics(TP, FP, FN)
            list_sst.append(sst)
            list_ppv.append(ppv)
            list_f1.append(f1)

        table_summary = pd.DataFrame({'TP':list_TP,
                                      'FP':list_FP,
                                      'FN':list_FN,
                                      'sensitivity':list_sst,
                                      'PPV':list_ppv,
                                      'F1':list_f1})
        return table_summary
