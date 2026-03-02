import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from cbct_simulate import CBCTSimulator

"""图像处理：以CT为基础，模拟生成CBCT
1. 读取nii.gz格式的CT为sitk对象
2. 执行模拟CBCT生成
4. 保存为2d.npy
"""


def main():
    mid_data_path = r'F:\2-MidData\sdfy_CBCT_to_CT\pelvic'
    final_data_path = r'F:\3-Data\2-sdfy_cbct_to_ct\pelvic'

    # ------------------------------
    sitk.ProcessObject_SetGlobalWarningDisplay(False)  # 不显示ITK警告
    patients = os.listdir(mid_data_path)
    patients = patients[1:]  # TODO 测试控制
    patient_num = len(patients)
    for i, patient in enumerate(patients):
        print('progress: %d/%d' % (i, patient_num))
        # 路径
        patient_mid_path = create_path(mid_data_path, patient)
        patient_final_path = create_path(final_data_path, patient)

        ct_path = os.path.join(patient_mid_path, 'CT.nii.gz')
        cbct_path = os.path.join(patient_mid_path, 'dCBCT.nii.gz')  # 仅用于可视化
        mask_path = os.path.join(patient_mid_path, 'Mask.nii.gz')

        # 数据读取
        ct = sitk.ReadImage(ct_path)
        cbct = sitk.ReadImage(cbct_path)
        mask = sitk.ReadImage(mask_path)
        # CBCT模拟
        simulator = CBCTSimulator()
        simulated_cbct = simulator.run(ct, mask)
        # plot_simulated_result(ct, cbct, simulated_cbct)  # Debug:绘图观察模拟效果

        # 存储
        simulated_cbct = sitk.Cast(simulated_cbct, sitk.sitkInt16)
        cbct_save_path = create_path(patient_final_path, 'sCBCT2')
        save_3d_img_to_2d_npy(simulated_cbct, cbct_save_path)
    # end for


def create_path(par_path: str, name: str):
    obj_path = os.path.join(par_path, name)
    if not os.path.exists(obj_path):
        os.mkdir(obj_path)
    # end if
    return obj_path


def save_3d_img_to_2d_npy(img, save_path):
    w, h, d = img.GetSize()
    img_3d = sitk.GetArrayFromImage(img)
    for i in range(d):
        img_2d = img_3d[i]
        save_path_2d = os.path.join(save_path, str(i).zfill(3) + '.npy')
        np.save(save_path_2d, img_2d)
    # end for


def plot_simulated_result(ct_sitk, cbct_sitk, simulated_cbct_sitk):
    ct3d = sitk.GetArrayFromImage(ct_sitk)
    cbct3d = sitk.GetArrayFromImage(cbct_sitk)
    simulated_cbct3d = sitk.GetArrayFromImage(simulated_cbct_sitk)
    n = ct3d.shape[0]

    for i in range(n):
        ct = ct3d[i]
        cbct = cbct3d[i]
        simulated_cbct = simulated_cbct3d[i]

        plt.figure(figsize=(15, 10))  # w, h
        manager = plt.get_current_fig_manager()
        # manager.window.wm_geometry(f"+100+10")  # w, h
        manager.window.move(10, 10)  # w, h
        plt.suptitle(str(i))
        plt.subplot(2, 3, 1)
        plt.imshow(ct, cmap='gray', vmin=-160, vmax=240)
        plt.title('CT')
        plt.axis('off')
        plt.subplot(2, 3, 2)
        plt.imshow(cbct, cmap='gray', vmin=-160, vmax=240)
        plt.title('CBCT')
        plt.axis('off')
        plt.subplot(2, 3, 3)
        plt.imshow(simulated_cbct, cmap='gray', vmin=-160, vmax=240)
        plt.title('simulatedCBCT')
        plt.axis('off')
        plt.subplot(2, 3, 5)
        diff = cbct - ct
        plt.imshow(diff, vmin=-50, vmax=50)
        mae, me = cal_mask_mae_me(ct, cbct)
        plt.title('Diff: CBCT - CT: ME:%.1f--MAE:%.1f' % (me, mae))
        plt.axis('off')
        plt.subplot(2, 3, 6)
        diff = simulated_cbct - ct
        plt.imshow(diff, vmin=-50, vmax=50)
        mae, me = cal_mask_mae_me(ct, simulated_cbct)
        plt.title('Diff: simulatedCBCT - CT: ME:%.1f--MAE:%.1f' % (me, mae))
        plt.axis('off')
        plt.show()
        plt.close()
    # end for i


def cal_mask_mae_me(ct, cbct):
    mask = ct > -900
    diff = cbct - ct
    import numpy.ma as ma
    diff = ma.masked_array(diff, mask == 0)
    mae = np.mean(np.abs(diff))
    me = np.mean(diff)
    return mae, me


if __name__ == '__main__':
    main()
