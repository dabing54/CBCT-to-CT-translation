import os
import shutil

"""从CT文件夹中将RS等文件分离出来"""

data_root_path = r'H:\sdfy_data\Pelvic2'
patients = os.listdir(data_root_path)
for patient in patients:
    patient_path = os.path.join(data_root_path, patient)
    ct_path = os.path.join(patient_path, 'CT')
    rt_path = os.path.join(patient_path, 'RT')
    if not os.path.exists(rt_path):
        os.mkdir(rt_path)
    items = os.listdir(ct_path)
    items = [item for item in items if not item.startswith('CT')]
    for item in items:
        item_path = os.path.join(ct_path, item)
        item_new_path = os.path.join(rt_path, item)
        shutil.move(item_path, item_new_path)