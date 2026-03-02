import os

"""重命名模型生成的结果，因为模型读取数据时首尾各去除了9层，保存结果时未考虑，此处需重命名补偿"""

assert False  # TODO 防止误操作，运行程序需要注释此句

data_root_path = r'F:\ProjectData\6-sCT4\out2'
series_list = os.listdir(data_root_path)
for series in series_list:
    series_root_path = os.path.join(data_root_path, series)
    patients = os.listdir(series_root_path)
    for patient in patients:
        patient_path = os.path.join(series_root_path, patient)
        items = os.listdir(patient_path)

        if items[0] == '009.npy':
            print('跳过：%s' % patient)
            continue

        num = len(items)
        if series == 'Unet' and num != 70:
            print(patient)
        for i in range(num-1, -1, -1):
            item = items[i]
            temp = str(i).zfill(3) + '.npy'
            assert item == str(i).zfill(3) + '.npy'
            new_item = str(i+9).zfill(3) + '.npy'
            item_path = os.path.join(patient_path, item)
            new_item_path = os.path.join(patient_path, new_item)
            assert not os.path.exists(new_item_path)
            os.rename(item_path, new_item_path)






