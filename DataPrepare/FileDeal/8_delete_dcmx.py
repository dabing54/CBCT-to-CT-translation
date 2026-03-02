import os

"""删除文件名以.dcmx结尾的文件
之前将CBCT首尾扫描不全的层面的文件名以.dcmx标识，
后来发现SimpleITK依然能够读取该文件，因此将其删除

后来发现形变配准会导致首尾层面缺失，或许此处不删除会缓解这一问题。
"""

data_root_path = r'G:\sdfy_data\Prostate'
patients = os.listdir(data_root_path)
for patient in patients:
    patient_path = os.path.join(data_root_path, patient)
    cbct_path = os.path.join(patient_path, 'CBCT')
    items = os.listdir(cbct_path)
    items = [item for item in items if item.endswith('.dcmx')]
    for item in items:
        item_path = os.path.join(cbct_path, item)
        # os.remove(item_path)  # TODO 使用时需注释，防止误操作
    # end for

