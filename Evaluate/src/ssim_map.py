import os
import numpy as np
import numpy.ma as ma
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt



def main():
    data_root_path_base = r'F:\3-Data\1-SynthRADTask2'
    data_root_path_gen = r'F:\ProjectData\6-sCT\SynthRADTask2\out'
    series_name = 'DDIM_sCBCT1_50step'
    # min_val = -160
    # max_val = 240

    min_val = -1024
    max_val = 1024

    display_min_val = -160
    display_max_val = 240

    # --------
    shift = -min_val
    display_min_val += shift
    display_max_val += shift
    # ----------------------------------
    series_path = os.path.join(data_root_path_gen, series_name)
    patients = os.listdir(series_path)
    # patients = patients[:1]
    patient_num = len(patients)
    for i, patient in enumerate(patients):
        print('Progress: %d/%d' % (i, patient_num))
        patient_path_gen = os.path.join(series_path, patient)
        patient_path_base = os.path.join(data_root_path_base, patient)
        patient_ct_path = os.path.join(patient_path_base, 'Full Dose Images')
        patient_cbct_path = os.path.join(patient_path_base, 'sCBCT1')
        patient_mask_path = os.path.join(patient_path_base, 'Mask')
        items = os.listdir(patient_path_gen)
        item_num = len(items)
        for j, item in enumerate(items):
            item_path2 = os.path.join(patient_path_gen, item)
            item_path1 = os.path.join(patient_ct_path, item)
            item_path3 = os.path.join(patient_cbct_path, item)
            mask_path = os.path.join(patient_mask_path, item)
            if not os.path.exists(item_path1):
                raise Exception('路径不存在：%s' %item_path1)
            img1 = np.load(item_path1)
            img2 = np.load(item_path2)
            mask = np.load(mask_path)
            img3 = np.load(item_path3)
            # print('img1: min: %d, max:%d' % (img1.min(), img1.max()))
            # print('img2: min: %d, max:%d' % (img2.min(), img2.max()))
            img1 = np.clip(img1, min_val, max_val)
            img2 = np.clip(img2, min_val, max_val)
            img3 = np.clip(img3, min_val, max_val)

            img1 += shift
            img2 += shift
            img3 += shift
            # ssim
            assert img1.min() >= 0
            assert img2.min() >= 0
            data_range = max_val - min_val
            _, ssim_map = structural_similarity(img2, img1, channel_axis=None, data_range=data_range, full=True)
            ssim_map = ma.masked_array(ssim_map, mask==0) if mask is not None else ssim_map
            ssim = ssim_map.mean()

            diff = ma.masked_array(img2 - img1, mask==0)

            # 绘图
            plt.figure(figsize=(19, 13))
            manager = plt.get_current_fig_manager()
            manager.window.wm_geometry("+00+00")
            plt.subplot(2, 3, 1)
            plt.title('label')
            plt.imshow(img1, cmap='gray', vmin=display_min_val, vmax=display_max_val)
            plt.subplot(2, 3, 2)
            plt.title('predict')
            plt.imshow(img2, cmap='gray', vmin=display_min_val, vmax=display_max_val)
            plt.subplot(2, 3, 3)
            plt.title('cbct')
            plt.imshow(img3, cmap='gray', vmin=display_min_val, vmax=display_max_val)
            plt.subplot(2, 3, 4)
            plt.title('predict-label')
            plt.imshow(diff, vmin=-30, vmax=30)
            plt.subplot(2, 3, 5)
            plt.title('ssim-%.3f' % ssim)
            plt.imshow(ssim_map, vmin=0.8, vmax=1, cmap='BuGn_r')  # Blues, Greens, BuGn

            plt.suptitle('%s-%d/%d : %s' % (patient, j, item_num, series_name))
            plt.tight_layout()
            plt.show()
            plt.close()
        # end for items
    # end for patients




if __name__ == '__main__':
    main()
