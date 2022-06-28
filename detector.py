import numpy as np
import matplotlib.pyplot as plt
import time
from utils.evaluator import Evaluator

peak_detector = Evaluator()


### Select database
# test_database = 'MIT_BIH'
# test_database = 'INCART'
# test_database = 'QTDB'
test_database = 'MIT_BIH_ST'
# test_database = 'European_ST_T'
# test_database = 'TELE'

'''
The current model was developed by training MIT_BIH, INCART, and QT databases.
If you test these databases, you will see the performance in the training set.
Cross-database testing is available when you test MIT_BIH_ST, European_ST_T, and TELE databases.
'''


### Run peak detection pipeline
print('Database ... {0}'.format(test_database))
start = time.time()
peak_detector.load(test_database)
peak_detector.find_peaks()
end = time.time()
elapsed = end-start
average_cost = elapsed/len(peak_detector.db_loading.metadata_patient)
print('Average elapsed time : {0:.2f}'.format(average_cost))


### Summary of model performance
table_summary = peak_detector.report_all()
table_summary.loc[table_summary.shape[0],:] = peak_detector.report_summary()
table_summary.index = peak_detector.db_loading.metadata_patient + ['Total']
table_summary = table_summary.round(decimals=4)

print('Summary of model performance')
print(table_summary)


### Visualize a specific ECGs
t_idx = 0
t_patient = table_summary.index[t_idx]
t_ecg = peak_detector.set_dict['ecg'][t_idx]
t_label = peak_detector.set_dict['label'][t_idx]
t_pred_TP = peak_detector.set_dict['pred_TP'][t_idx]
t_pred_FP = peak_detector.set_dict['pred_FP'][t_idx]
t_pred_FN = peak_detector.set_dict['pred_FN'][t_idx]
t_xtick = np.arange(t_ecg.shape[0])/360

plt.plot(t_xtick, t_ecg, color='black')
plt.plot(t_xtick[t_pred_TP], [t_ecg[x] for x in t_pred_TP], 'o', color='green')
plt.plot(t_xtick[t_pred_FP], [t_ecg[x] for x in t_pred_FP], '*', color='red')
plt.plot(t_xtick[t_pred_FN], [t_ecg[x] for x in t_pred_FN], '*', color='blue')
plt.title('Database {}, Patient {}'.format(test_database, t_patient))
plt.xlabel('Time (s)')
plt.ylabel('Voltage (mV)')
plt.show()
