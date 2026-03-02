import os
import pydicom
import numpy as np
from datetime import datetime

data_root_path = r'H:\sdfy_data\Pelvic'
patients = os.listdir(data_root_path)
patients_num = len(patients)
out = np.zeros(patients_num)
for i, patient in enumerate(patients):
    patient_path = os.path.join(data_root_path, patient)
    # path
    ct_path = os.path.join(patient_path, 'CT')
    cbct_path = os.path.join(patient_path, 'CBCT')
    item = os.listdir(ct_path)[0]
    ct_item_path = os.path.join(ct_path, item)
    cbct_item_path = os.path.join(cbct_path, item)
    # CT
    dcm = pydicom.dcmread(ct_item_path)
    date1 = dcm['ContentDate']
    
    # CBCT
    dcm = pydicom.dcmread(cbct_item_path)
    data2 = dcm['ContentDate']
    # diff
    diff = 1
    out[i] = diff
# end for
p0 = np.min(out)
p25 = 1
# p50 =
# p75 =
# p100 =
# std =