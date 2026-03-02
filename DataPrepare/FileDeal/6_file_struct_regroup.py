import os
import shutil

"""文件结构重组
原结构类型：root-series-patient
新结构类型：root-patient-series
"""

data_root_path = r'H:\sdfy_data\Pelvic2'
ct_root_path = os.path.join(data_root_path, 'CT')
cbct_root_path = os.path.join(data_root_path, 'CBCT')

patients1 = os.listdir(ct_root_path)
patients2 = os.listdir(cbct_root_path)
assert patients1 == patients2

for patient in patients1:
    patient_ct_path = os.path.join(ct_root_path, patient)
    patient_cbct_path = os.path.join(cbct_root_path, patient)
    new_patient_path = os.path.join(data_root_path, patient)
    if not os.path.exists(new_patient_path):
        os.mkdir(new_patient_path)
    # end if
    new_patient_ct_path = os.path.join(new_patient_path, "CT")
    new_patient_cbct_path = os.path.join(new_patient_path, 'CBCT')
    shutil.move(patient_ct_path, new_patient_ct_path)
    shutil.move(patient_cbct_path, new_patient_cbct_path)
