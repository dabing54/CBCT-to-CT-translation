import os
import shutil

"""dcm文件分组
用于对Varian导出的dcm文件进行分组。
根据文件名读取患者id进行分组
"""

data_root_path = r'H:\sdfy_data\Pelvic2\CBCT'
items = os.listdir(data_root_path)
for item in items:
    patient_id = item.split('.')[1]
    patient_path = os.path.join(data_root_path, patient_id)
    if not os.path.exists(patient_path):
        os.mkdir(patient_path)
    item_path = os.path.join(data_root_path, item)
    item_new_path = os.path.join(patient_path, item)
    shutil.move(item_path, item_new_path)
# end for

