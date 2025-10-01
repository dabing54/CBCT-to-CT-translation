import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    data_root_path_base = r'F:\3-Data\1-SynthRADTask2'
    data_root_path_gen = r'F:\ProjectData\6-sCT3\SynthRADTask2\out'
    save_root_path = r'E:\DaBing54\Desktop\out'
    patient = '2PA047'  # 2PA047  -- 020.npy  032.npy
    item = '032.npy'
    series_name = 'FlowMatch_sCBCT1_50step'

    dpi = 600
    shrink = 0.5

    row_limit1 = 65  # 去除留白的行
    row_limit2 = 205

    min_val = -1024
    max_val = 1024

    display_min_val = -160
    display_max_val = 240

    series_path = os.path.join(data_root_path_gen, series_name)
    patient_path_gen = os.path.join(series_path, patient)
    patient_path_base = os.path.join(data_root_path_base, patient)
    patient_ct_path = os.path.join(patient_path_base, 'Full Dose Images')
    ct_path = os.path.join(patient_ct_path, item)
    patient_cbct_path = os.path.join(patient_path_base, 'sCBCT1')
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
    sct = sct[row_limit1: row_limit2, :]
    sct = np.clip(sct, min_val, max_val)

    # body外置0
    ct[mask == 0] = min_val
    cbct[mask == 0] = min_val
    sct[mask == 0] = min_val

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    # 1.
    plt.figure(figsize=(5, 5))
    plt.imshow(cbct, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')
    save_path = os.path.join(save_root_path, 'cbct.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    # 2.
    plt.figure(figsize=(5, 5))
    plt.imshow(ct, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')
    save_path = os.path.join(save_root_path, 'ct.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    # 3.
    plt.figure(figsize=(5, 5))
    plt.imshow(sct, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')
    save_path = os.path.join(save_root_path, 'sct.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    # 4
    error = sct - ct
    plt.figure(figsize=(5, 5))
    plt.imshow(error, cmap='seismic', vmin=-300, vmax=300)
    plt.axis('off')
    plt.colorbar(shrink=shrink)
    save_path = os.path.join(save_root_path, 'sct HU error map.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    # 5
    points_x = ct[mask == 1]
    points_y = sct[mask == 1]
    plt.figure(figsize=(5, 5))
    plt.plot([-1024, 1024], [-1024, 1024], color='darkgreen', linestyle='--')
    plt.scatter(points_x, points_y, s=1)
    ax = plt.gca()
    ax.set_xlim(-1024, 1024)
    ax.set_ylim(-1024, 1024)
    save_path = os.path.join(save_root_path, 'sct HU error scatter')
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    # 6
    plt.figure(figsize=(5, 5))
    from skimage.metrics import structural_similarity
    import numpy.ma as ma
    _, ssim = structural_similarity(sct + 1024, ct + 1024, channel_axis=None, data_range=2048, full=True)
    ssim = ma.masked_array(ssim, mask == 0) if mask is not None else ssim
    plt.imshow(ssim, vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar(shrink=shrink)
    save_path = os.path.join(save_root_path, 'sct ssim map')
    plt.savefig(save_path, dpi=dpi)
    plt.close()

    # 7
    error = cbct - ct
    plt.figure(figsize=(5, 5))
    plt.imshow(error, cmap='seismic', vmin=-300, vmax=300)
    plt.axis('off')
    plt.colorbar(shrink=shrink)
    save_path = os.path.join(save_root_path, 'cbct HU error map.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    # 8
    points_x = ct[mask == 1]
    points_y = cbct[mask == 1]
    plt.figure(figsize=(5, 5))
    plt.plot([-1024, 1024], [-1024, 1024], color='darkgreen', linestyle='--')
    plt.scatter(points_x, points_y, s=1)
    ax = plt.gca()
    ax.set_xlim(-1024, 1024)
    ax.set_ylim(-1024, 1024)
    save_path = os.path.join(save_root_path, 'cbct HU error scatter')
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    # 9
    plt.figure(figsize=(5, 5))
    from skimage.metrics import structural_similarity
    import numpy.ma as ma
    _, ssim = structural_similarity(cbct + 1024, ct + 1024, channel_axis=None, data_range=2048, full=True)
    ssim = ma.masked_array(ssim, mask == 0) if mask is not None else ssim
    plt.imshow(ssim, vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar(shrink=shrink)
    save_path = os.path.join(save_root_path, 'cbcct ssim map')
    plt.savefig(save_path, dpi=dpi)
    plt.close()

if __name__ == '__main__':
    main()
