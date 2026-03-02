import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt

"""逐个查看一个序列的图像"""

data_root_path = r'H:\sdfy_data\CBCT'
vmin = -160  # 绘图显示最小值
vmax = 240  # 绘图显示最大值
reverse = True  # TODO 绘图时，是否层面反序
#
patients = os.listdir(data_root_path)
patients = patients[41:]  # TODO 用于控制查看的患者id
for patient in patients:
    patient_path = os.path.join(data_root_path, patient)
    items = os.listdir(patient_path)
    if reverse:
        items.reverse()
    for item in items:
        item_path = os.path.join(patient_path, item)
        dcm = pydicom.dcmread(item_path)
        img = dcm.pixel_array
        rescale_slope = dcm.get("RescaleSlope", 1)  # 无字段则默认1
        rescale_intercept = dcm.get("RescaleIntercept", 0)  # 无字段则默认0
        img = img * rescale_slope + rescale_intercept  # 应用转换：img * Slope + Intercept
        # 绘图
        plt.figure(figsize=(8, 8))
        mngr = plt.get_current_fig_manager()
        mngr.window.wm_geometry("+300+0")
        plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.title(patient + '__' + item)
        plt.show()
    # end for
# end for


