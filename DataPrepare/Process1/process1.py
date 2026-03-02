import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from dicom_read import get_dcm_img, get_dcm_img_mask
from registration import rigid_registration, apply_transform, deform_registration

"""图像处理：CBCT与CT配准，并裁剪多余区域
1. 读取dcm形式的CT，RS的body区域，转换为sitk形式的img, mask
2. 读取dcm形式的CBCT，转换为sitk形式
3. 保存为3d.nii
4. 保存为2d.npy
"""

def main():
    data_root_path = r'H:\sdfy_data\Pelvic'  # F:\TestData
    mid_data_path = r'F:\2-MidData\sdfy_CBCT_to_CT\pelvic'
    final_data_path = r'F:\3-Data\2-sdfy_cbct_to_ct\pelvic'
    roi_names = ['Body', 'BODY', 'External', 'patient', 'Patient']  # Body区域命名
    max_connected_domain = '3d'  # 对3D mask保留最大连通域  'none', '3d', '2d'

    # ------------------------------
    sitk.ProcessObject_SetGlobalWarningDisplay(False)  # 不显示ITK警告
    patients = os.listdir(data_root_path)
    patients = patients[90:]  # TODO 测试控制
    patient_num = len(patients)
    for i, patient in enumerate(patients):
        print('progress: %d/%d' % (i, patient_num))
        patient_path = os.path.join(data_root_path, patient)
        patient_mid_path = create_path(mid_data_path, patient)
        patient_final_path = create_path(final_data_path, patient)
        is_empty = not os.listdir(patient_final_path)
        if not is_empty:
            print('跳过患者：%s' % patient)
        else:
            deal_patient(patient_path, patient_mid_path, patient_final_path, roi_names, max_connected_domain)

def deal_patient(patient_path, patinet_mid_path, patient_final_path, roi_names, max_connected_domain,):
    # 路径
    cbct_path = os.path.join(patient_path, 'CBCT')
    ct_path = os.path.join(patient_path, 'CT')
    rs_path = get_rs_path(patient_path)
    # 读取、裁剪
    cbct = get_dcm_img(cbct_path)
    ct, mask = get_dcm_img_mask(ct_path, rs_path, roi_names, max_connected_domain)
    ct, mask = crop_img_by_mask(ct, mask)
    check_direction(ct, mask)
    # 刚性配准：CT -> CBCT  可以去除CT多余的层面
    cbct = reset_origin(ct, cbct)  # 重置原点，防止刚性配准差异太远出错
    ct, transform_params = rigid_registration(cbct, ct, is_3d=True, return_transform=True)
    mask = apply_transform(mask, transform_params, is_mask=True)
    # 根据mask裁剪掉周围空白区域
    ct, mask, cbct = crop_img_by_mask(ct, mask, img2=cbct)
    # 重采样
    ct, mask, cbct = resample_img_mask(ct, mask, obj_size=256, img2=cbct)
    # 形变配准：CBCT -> CT，据说效果更好
    deformed_cbct = deform_registration(ct, cbct, is_3d=True)
    # plot_registration_result(ct, cbct, deformed_cbct)  # todo Debug:绘图观察形变配准效果

    # 保存前格式转换
    ct = sitk.Cast(ct, sitk.sitkInt16)
    cbct = sitk.Cast(cbct, sitk.sitkInt16)
    deformed_cbct = sitk.Cast(deformed_cbct, sitk.sitkInt16)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    # 保存，3d形式
    ct_save_path = os.path.join(patinet_mid_path, 'CT.nii.gz')
    cbct_save_path = os.path.join(patinet_mid_path, 'CBCT.nii.gz')
    deformed_cbct_save_path = os.path.join(patinet_mid_path, 'dCBCT.nii.gz')
    mask_save_path = os.path.join(patinet_mid_path, 'Mask.nii.gz')
    sitk.WriteImage(ct, ct_save_path)
    sitk.WriteImage(cbct, cbct_save_path)
    sitk.WriteImage(deformed_cbct, deformed_cbct_save_path)
    sitk.WriteImage(mask, mask_save_path)

    # 保存，2d形式
    ct_save_path = create_path(patient_final_path, 'CT')
    deformed_cbct_save_path = create_path(patient_final_path, 'dCBCT')
    mask_save_path = create_path(patient_final_path, 'Mask')
    save_3d_img_to_2d_npy(ct, ct_save_path)
    save_3d_img_to_2d_npy(deformed_cbct, deformed_cbct_save_path)
    save_3d_img_to_2d_npy(mask, mask_save_path)

def resample_img_mask(img, mask, obj_size:int, img2=None):
    """同时对img和mask进行重采样,层数不变，宽高的最长者重采样为目标尺寸"""
    space = img.GetSpacing()  # w, h, d
    size = img.GetSize()
    # 计算新size、space
    scale = obj_size / np.max(size[:2])
    new_size_hw = np.round(scale * np.array(size[:2])).astype(int)
    new_size = np.append(new_size_hw, size[2]).tolist()
    new_space = np.array(space)
    new_space[:2] = new_space[:2] / scale
    # 重采样
    img = resample(img, new_size, new_space, is_mask=None)
    mask = resample(mask, new_size, new_space, is_mask=True)
    if img2 is not None:
        img2 = resample(img2, new_size, new_space, is_mask=None)
    return img, mask, img2

def resample(img, new_size, new_space, is_mask=None):
    """层数不变，宽高的最长者重采样为目标尺寸"""
    direction = img.GetDirection()
    origin = img.GetOrigin()
    resamper = sitk.ResampleImageFilter()
    resamper.SetReferenceImage(img)
    resamper.SetSize(new_size)
    resamper.SetOutputSpacing(new_space)
    resamper.SetOutputOrigin(origin)
    resamper.SetOutputDirection(direction)
    resamper.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resample_method = sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline
    default_vale = 0 if is_mask else -1024
    out_pixel_type = sitk.sitkUInt8 if is_mask else sitk.sitkInt16
    resamper.SetOutputPixelType(out_pixel_type)
    resamper.SetDefaultPixelValue(default_vale)
    resamper.SetInterpolator(resample_method)
    img = resamper.Execute(img)
    return img


def create_path(par_path:str, name:str):
    obj_path = os.path.join(par_path, name)
    if not os.path.exists(obj_path):
        os.mkdir(obj_path)
    # end if
    return obj_path

def save_3d_img_to_2d_nii(img, save_path):
    w, h, d = img.GetSize()
    img_3d = sitk.GetArrayFromImage(img)
    space = img.GetSpacing()[:2]
    for i in range(d):
        img_2d = sitk.GetImageFromArray(img_3d[i])
        img_2d.SetSpacing(space)
        save_path_2d = os.path.join(save_path, str(i).rjust(3, '0') + '.nii')
        sitk.WriteImage(img_2d, save_path_2d)
    # end for

def save_3d_img_to_2d_npy(img, save_path):
    w, h, d = img.GetSize()
    img_3d = sitk.GetArrayFromImage(img)
    for i in range(d):
        img_2d = img_3d[i]
        save_path_2d = os.path.join(save_path, str(i).zfill(3) + '.npy')
        np.save(save_path_2d, img_2d)
    # end for

def get_rs_path(patient_path):
    rt_path = os.path.join(patient_path, 'RT')
    items = os.listdir(rt_path)
    items = [item for item in items if item.startswith('RS')]
    assert len(items) ==  1
    rs_path = os.path.join(rt_path, items[0])
    return rs_path

def crop_img_by_mask(img, mask, gap=10, img2=None):
    """输入、输出均为sitk类型，层内裁剪, 层数也裁剪"""
    mask_3d = sitk.GetArrayFromImage(mask)
    # 层内边界
    mask_2d = np.sum(mask_3d, axis=0) > 0  # h, w
    line = np.sum(mask_2d, axis=0) > 0
    wms, wme = cal_remove_margin(line, gap)
    line = np.sum(mask_2d, axis=1) > 0
    hms, hme = cal_remove_margin(line, gap)
    # 层数边界
    line = np.sum(mask_3d, axis=(1, 2)) > 0
    dms, dme = cal_remove_margin(line, gap=0)

    margin_s = (wms, hms, dms)
    margin_e = (wme, hme, dme)

    # sitk.Crop函数是指定需要去除的部分，sitk.Crop(img, 去除1，去除2)
    # Extract函数指定需要保留的部分，RegionOfInterest也指定需要保留的部分
    img = sitk.Crop(img, margin_s, margin_e)
    mask = sitk.Crop(mask, margin_s, margin_e)
    if img2 is not None:
        img2 = sitk.Crop(img2, margin_s, margin_e)
        return img, mask, img2
    return img, mask

def cal_remove_margin(line, gap):
    count = len(line)
    idx = np.where(line > 0)[0]
    if len(idx) == 0:
        margin_s, margin_e = 0, 0
    else:
        idx_s = np.min(idx)
        idx_e = np.max(idx)
        margin_s = idx_s
        margin_e = count - idx_e - 1
        margin_s = max(0, margin_s - gap)
        margin_e = max(0, margin_e - gap)
        margin_s, margin_e = int(margin_s), int(margin_e)
    return margin_s, margin_e

def debug_plot_3d_img(img_list, idx=0, cmap='viridis'):
    """输入为sitk对象列表[img1, img2, ...]"""
    num = len(img_list)
    plt.figure(figsize=(15, 5))  # w, h
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry(f"+100+10")  # w, h
    for i in range(num):
        arr = sitk.GetArrayFromImage(img_list[i])
        plt.subplot(1, num, i + 1)
        plt.axis('off')
        plt.imshow(arr[idx], cmap)
    plt.suptitle(idx)
    plt.show()
    plt.close()

def debug_plot_3d_img_multi(img_list, cmap='viridis'):
    """输入为sitk对象列表[img1, img2, ...]"""
    slice_num = img_list[0].GetDepth()
    for idx in range(slice_num):
        debug_plot_3d_img(img_list, idx, cmap)

def reset_origin(fixed_img, moving_img):
    moving_img.SetOrigin(fixed_img.GetOrigin())
    return moving_img

def check_direction(img1, img2):
    direction1 = img1.GetDirection()
    direction2 = img2.GetDirection()
    assert direction1 == direction2

def plot_registration_result(ct_sitk, cbct_sitk, deformed_cbct_sitk):
    ct3d = sitk.GetArrayFromImage(ct_sitk)
    cbct3d = sitk.GetArrayFromImage(cbct_sitk)
    deformed_cbct3d = sitk.GetArrayFromImage(deformed_cbct_sitk)
    n = ct3d.shape[0]

    red_cmap = LinearSegmentedColormap.from_list("black2red", [(0, 0, 0), (1, 0, 0)], N=256)
    # 方案2：绿色伪彩（可选）
    green_cmap = LinearSegmentedColormap.from_list("black2green", [(0, 0, 0), (0, 1, 0)], N=256)

    for i in range(n):
        ct = ct3d[i]
        cbct = cbct3d[i]
        deformed_cbct = deformed_cbct3d[i]

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
        plt.imshow(deformed_cbct, cmap='gray', vmin=-160, vmax=240)
        plt.title('deformedCBCT')
        plt.axis('off')
        plt.subplot(2, 3, 5)
        plt.imshow(ct, cmap=green_cmap, vmin=-160, vmax=240, alpha=0.5)
        plt.imshow(cbct, cmap=red_cmap, vmin=-160, vmax=240, alpha=0.5)
        plt.title('Fused: CT(Green)+ CBCT(Red)')
        plt.axis('off')
        plt.subplot(2, 3, 6)
        plt.imshow(ct, cmap=green_cmap, vmin=-160, vmax=240, alpha=0.5)
        plt.imshow(deformed_cbct, cmap=red_cmap, vmin=-160, vmax=240, alpha=0.5)
        plt.title('Fused: CT(Green)+ deformedCBCT(Red)')
        plt.axis('off')
        plt.show()
        plt.close()
    # end for i

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
        plt.imshow(diff, vmin=-100, vmax=100)
        mae, me = cal_mask_mae_me(ct, cbct)
        plt.title('Diff: CBCT - CT: ME:%.1f--MAE:%.1f' % (me, mae))
        plt.axis('off')
        plt.subplot(2, 3, 6)
        diff = simulated_cbct - ct
        plt.imshow(diff,vmin=-100, vmax=100)
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
    diff = ma.masked_array(diff, mask==0)
    mae = np.mean(np.abs(diff))
    me = np.mean(diff)
    return mae, me


if __name__ == '__main__':
    main()
