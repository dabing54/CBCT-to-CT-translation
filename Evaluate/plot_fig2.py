import os
import numpy as np
import matplotlib.pyplot as plt
from tools import center_crop


def main():
    data_root_path_base = r'F:\3-Data\2-sdfy_cbct_to_ct\pelvic'
    data_root_path_gen = r'F:\ProjectData\6-sCT4\out'
    patient = '252406'  # 2PA047  -- 020.npy  032.npy   # 252505  252406
    item = '040.npy'  # 252456-35   253142-70-66  252930-21
    series_name = 'Flow-20step'  # Flow-20step-dCBCT   # 修改此处需同步修改line32

    # Flow好：252505-18

    row_limit1 = 0  # 去除留白的行
    row_limit2 = 256

    row_num = 50

    min_val = -1024
    max_val = 1024

    display_min_val = -160
    display_max_val = 240

    series_path = os.path.join(data_root_path_gen, series_name)
    patient_path_gen = os.path.join(series_path, patient)
    patient_path_base = os.path.join(data_root_path_base, patient)
    patient_ct_path = os.path.join(patient_path_base, 'CT')
    ct_path = os.path.join(patient_ct_path, item)
    patient_cbct_path = os.path.join(patient_path_base, 'sCBCT2')  # sCBCT2  dCBCT
    cbct_path = os.path.join(patient_cbct_path, item)
    patient_mask_path = os.path.join(patient_path_base, 'Mask')
    mask_path = os.path.join(patient_mask_path, item)
    base_item_path = os.path.join(patient_path_gen, item)
    if not os.path.exists(ct_path):
            raise Exception('路径不存在：%s' % ct_path)

    ct = np.load(ct_path)
    ct = ct[row_limit1: row_limit2, :]
    ct = np.clip(ct, min_val, max_val)

    cbct = np.load(cbct_path)
    cbct = cbct[row_limit1: row_limit2, :]
    cbct = np.clip(cbct, min_val, max_val)

    mask = np.load(mask_path)
    mask = mask[row_limit1: row_limit2, :]

    sct = np.load(base_item_path)
    sct = center_crop(sct, ct)
    sct = sct[row_limit1: row_limit2, :]
    sct = np.clip(sct, min_val, max_val)

    # body外置0
    ct[mask == 0] = min_val
    cbct[mask == 0] = min_val
    sct[mask == 0] = min_val

    plt.figure(figsize=(15, 10))
    plt.rcParams['font.family'] = 'Times New Roman'
    mngr = plt.get_current_fig_manager()
    mngr.window.wm_geometry("+0+100")

    # 1
    plt.subplot2grid((3, 3), (0, 0))
    plt.imshow(cbct, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')
    plt.title('CBCT')
    # 2.
    plt.subplot2grid((3, 3), (0, 1))
    plt.imshow(ct, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')
    plt.title('CT')
    # 3.
    plt.subplot2grid((3, 3), (0, 2))
    plt.imshow(sct, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')
    plt.title('sCT')

    # 4
    plt.subplot2grid((3, 3), (1, 1))
    error = sct - ct
    plt.imshow(error, cmap='seismic', vmin=-300, vmax=300)
    plt.axis('off')
    plt.colorbar(shrink=0.5)
    error = np.ma.masked_array(error, mask==0)
    me_mean = np.mean(error)
    me_std = np.std(error)
    error = np.abs(error)
    mae_mean = np.mean(error)
    mae_std = np.std(error)
    plt.title('(sCT - CT): me:%.2f+%.2f.mae:%.2f+%.2f' % (me_mean , me_std, mae_mean, mae_std))

    # 5
    plt.subplot2grid((3, 3), (1, 0))  # , rowspan=1, colspan=2
    points_x = ct[mask == 1]
    points_y = sct[mask == 1]
    plt.plot([-1024, 1024], [-1024, 1024], color='darkgreen', linestyle='--')
    plt.scatter(points_x, points_y, s=1)
    plt.title('HU error scatter: sCT')
    ax = plt.gca()
    ax.set_xlim(-1024, 1024)
    ax.set_ylim(-1024, 1024)
    ax.set_position([0.13, 0.43, 0.12, 0.12])

    # 6
    plt.subplot2grid((3, 3), (1, 2))
    from skimage.metrics import structural_similarity
    import numpy.ma as ma
    _, ssim = structural_similarity(sct + 1024, ct + 1024, channel_axis=None, data_range=2048, full=True)
    assert mask is not None
    ssim = ma.masked_array(ssim, mask == 0) if mask is not None else ssim
    plt.imshow(ssim, vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar(shrink=0.6)
    ssim_mean = np.mean(ssim)
    ssim_std = np.std(ssim)
    plt.title('SSIM map: sCT:ssim:%.3f+%.3f' % (ssim_mean, ssim_std))

    # 7
    plt.subplot2grid((3, 3), (2, 1))
    error = cbct - ct
    plt.imshow(error, cmap='seismic', vmin=-300, vmax=300)
    plt.axis('off')
    plt.colorbar(shrink=0.7)
    # plt.title('HU error map (CBCT - CT)')
    error = np.ma.masked_array(error, mask == 0)
    me_mean = np.mean(error)
    me_std = np.std(error)
    error = np.abs(error)
    mae_mean = np.mean(error)
    mae_std = np.std(error)
    plt.title('(CBCT - CT): me:%.2f+%.2f.mae:%.2f+%.2f' % (me_mean, me_std, mae_mean, mae_std))

    # 8
    plt.subplot2grid((3, 3), (2, 0))  # , rowspan=1, colspan=2
    points_x = ct[mask == 1]
    points_y = cbct[mask == 1]
    plt.plot([-1024, 1024], [-1024, 1024], color='darkgreen', linestyle='--')
    plt.scatter(points_x, points_y, s=1)
    plt.title('HU error scatter: CBCT')
    # plt.axis('off')
    ax = plt.gca()
    ax.set_xlim(-1024, 1024)
    ax.set_ylim(-1024, 1024)
    ax.set_position([0.13, 0.36, 0.1, 0.1])

    # 9
    plt.subplot2grid((3, 3), (2, 2))
    from skimage.metrics import structural_similarity
    import numpy.ma as ma
    _, ssim = structural_similarity(cbct + 1024, ct + 1024, channel_axis=None, data_range=2048, full=True)
    assert mask is not None
    ssim = ma.masked_array(ssim, mask == 0) if mask is not None else ssim
    plt.imshow(ssim, vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar(shrink=0.7)
    # plt.title('SSIM map: CBCT')
    ssim_mean = np.mean(ssim)
    ssim_std = np.std(ssim)
    plt.title('SSIM map: sCT:ssim:%.3f+%.3f' % (ssim_mean, ssim_std))

    plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, wspace=0.1, hspace=0.1)

    plt.show()

if __name__ == '__main__':
    main()
