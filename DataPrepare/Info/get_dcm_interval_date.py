import os
import pydicom
import numpy as np
from datetime import datetime

data_root_path = r'H:\sdfy_data\Pelvic'
patients = os.listdir(data_root_path)
patients_num = len(patients)
out = np.zeros(patients_num)
for i, patient in enumerate(patients):
    print('Progress: %d/%d' % (i, patients_num))
    patient_path = os.path.join(data_root_path, patient)
    # path
    ct_path = os.path.join(patient_path, 'CT')
    cbct_path = os.path.join(patient_path, 'CBCT')
    item = os.listdir(ct_path)[0]
    ct_item_path = os.path.join(ct_path, item)
    cbct_item_path = os.path.join(cbct_path, item)
    # CT
    date_format = '%Y%m%d'
    dcm = pydicom.dcmread(ct_item_path)
    date_str1 = dcm['ContentDate'].value
    date1 = datetime.strptime(date_str1, date_format)
    # CBCT
    dcm = pydicom.dcmread(cbct_item_path)
    date_str2 = dcm['ContentDate'].value
    date2 = datetime.strptime(date_str2, date_format)
    # diff
    diff = date2 - date1
    diff_d = diff.days
    out[i] = diff_d
# end for
out = out[np.bitwise_and(out>0, out<24)]  # TODO

p0 = np.min(out)
p25 = np.percentile(out, 25)
p50 = np.percentile(out, 50)
p75 = np.percentile(out, 75)
p100 = np.max(out)
mean = np.mean(out)
std = np.std(out)

print(out)
print('P0: %d' % p0)
print('P25: %d' % p25)
print('P50: %d' % p50)
print('P75: %d' % p75)
print('P100: %d' % p100)
print('mean: %.2f' % mean)
print('std: %.2f' % std)