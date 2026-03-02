import os

"""用于修改文件名
1. 将层数标识统一为3位数字，解决文件排序时9.dcm排在10.dcm后面的问题。依据Varian导出的文件名格式处理
"""

data_root_path = r'H:\sdfy_data\Pelvic2\CBCT'

patients = os.listdir(data_root_path)
for patient in patients:
    patient_path = os.path.join(data_root_path, patient)
    items = os.listdir(patient_path)
    items = [item for item in items if item.startswith('CT')]  # RS不处理
    for item in items:
        parts = item.split('.')
        assert len(parts) == 4
        parts[2] = parts[2].zfill(3)
        new_item = '.'.join(parts)
        item_path = os.path.join(patient_path, item)
        new_item_path = os.path.join(patient_path, new_item)
        os.rename(item_path, new_item_path)
    # end for
# end for




