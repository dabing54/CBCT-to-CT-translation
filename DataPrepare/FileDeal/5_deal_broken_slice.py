import os

"""根据自定义的起始层面和终止层面，将范围外的层面文件名后缀加个x"""

data_root_path = r'G:\sdfy_data\Prostate\CBCT'
s_list = [4, 5, 4, 4, 5, 5, 5, 4, 4, 6,  # 第一个完整的层面的名字
          5, 4, 4, 5, 5, 5, 5, 5, 4, 4,
          5, 4, 4, 5, 4, 6, 5, 5, 5, 5,
          5, 5, 5, 5, 5, 5, 5, 5, 4, 5,
          5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
          5, 5, 4, 5, 5, 4, 6, 4, 5, 5,
          5, 5, 7, 4, 5]
e_list = [84, 85, 85, 85, 85, 85, 84, 86, 84, 85,
          85, 86, 86, 84, 84, 84, 85, 82, 85, 85,
          85, 85, 85, 83, 86, 84, 84, 84, 84, 85,
          84, 85, 85, 83, 85, 84, 85, 85, 84, 84,
          85, 85, 85, 85, 85, 83, 84, 84, 84, 84,
          68, 84, 83, 84, 85, 85, 84, 85, 85, 84,
          85, 85, 72, 85, 85]

patients = os.listdir(data_root_path)
assert len(e_list) == len(s_list)
assert len(s_list) == len(patients)
for i, patient in enumerate(patients):
    patient_path = os.path.join(data_root_path, patient)
    s = s_list[i]
    e = e_list[i]
    items = os.listdir(patient_path)
    for item in items:
        idx = int(item.split('.')[2])
        if idx < s or idx > e:
            item_new = item + 'x'
            new_item_path = os.path.join(patient_path, item_new)
            item_path = os.path.join(patient_path, item)
            os.rename(item_path, new_item_path)
        # end if
    # end for
# end for

