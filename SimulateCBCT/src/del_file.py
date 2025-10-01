import os
import shutil

data_root_path = r'F:\3-Data\1-SynthRADTask2'
patients = os.listdir(data_root_path)
for patient in patients:
    patient_path = os.path.join(data_root_path, patient)
    items = os.listdir(patient_path)
    for item in items:
        if item in ['sCBCT', 'sCBCT2', 'sCBCT3']:
            item_path = os.path.join(patient_path, item)
            shutil.rmtree(item_path)