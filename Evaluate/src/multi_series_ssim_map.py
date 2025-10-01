import os
import numpy as np
import numpy.ma as ma
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt


def main():
    data_root_path_base = r'F:\3-Data\1-SynthRADTask2'
    data_root_path_gen = r'F:\ProjectData\6-sCT\SynthRADTask2\out'
    series_name_list = ['Flow_sCBCT1_2step', 'Flow_sCBCT1_10step', 'Flow_sCBCT1_50step']
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
    series_num = len(series_name_list)
    # ----------------------------------
    series_path = os.path.join(data_root_path_gen, series_name_list[0])
    patients = os.listdir(series_path)
    # patients = patients[:1]
    patient_num = len(patients)
    for i, patient in enumerate(patients):
        print('Progress: %d/%d' % (i, patient_num))
        patient_path_gen = os.path.join(series_path, patient)
        patient_path_base = os.path.join(data_root_path_base, patient)
        patient_ct_path = os.path.join(patient_path_base, 'Full Dose Images')
        patient_mask_path = os.path.join(patient_path_base, 'Mask')
        patient_cbct_path = os.path.join(patient_path_base, 'sCBCT1')
        items = os.listdir(patient_path_gen)
        item_num = len(items)
        for j, item in enumerate(items):
            ct_path = os.path.join(patient_ct_path, item)
            cbct_path = os.path.join(patient_cbct_path, item)
            mask_path = os.path.join(patient_mask_path, item)
            base_item_path = os.path.join(patient_path_gen, item)
            if not os.path.exists(ct_path):
                raise Exception('路径不存在：%s' %ct_path)

            ct = np.load(ct_path)
            cbct = np.load(cbct_path)
            mask = np.load(mask_path)

            ct = np.clip(ct, min_val, max_val)
            cbct = np.clip(cbct, min_val, max_val)

            ct += shift
            cbct += shift

            assert ct.min() >= 0
            data_range = max_val - min_val

            # 绘图
            plt.figure(figsize=(19, 13))
            manager = plt.get_current_fig_manager()
            manager.window.wm_geometry("+00+00")
            col = series_num + 1
            plt.subplot(3, col, 1)
            plt.title('label')
            plt.imshow(ct, cmap='gray', vmin=display_min_val, vmax=display_max_val)
            plt.subplot(3, col, col+1)
            plt.title('cbct')
            plt.imshow(cbct, cmap='gray', vmin=display_min_val, vmax=display_max_val)
            plt.subplot(3, col, 2 * col + 1)
            plt.title('diff')
            diff = ma.masked_array(cbct - ct, mask == 0)
            plt.imshow(diff, vmin=-30, vmax=30)

            # sCT
            for k, series_name in enumerate(series_name_list):
                if k != 0:
                    item_path = base_item_path.replace(series_name_list[0], series_name_list[k])
                else:
                    item_path = base_item_path
                # end if

                sCT = np.load(item_path)
                sCT = np.clip(sCT, min_val, max_val)
                sCT += shift

                # ssim
                assert sCT.min() >= 0
                _, ssim_map = structural_similarity(sCT, ct, channel_axis=None, data_range=data_range, full=True)
                ssim_map = ma.masked_array(ssim_map, mask == 0) if mask is not None else ssim_map
                ssim = ssim_map.mean()

                diff = ma.masked_array(sCT - ct, mask == 0)

                # 绘图
                plt.subplot(3, col, k+2)
                plt.title(series_name)  # predict
                plt.imshow(sCT, cmap='gray', vmin=display_min_val, vmax=display_max_val)
                plt.subplot(3, col, col+k+2)
                plt.title('ssim-%.3f' % ssim)
                plt.imshow(ssim_map, vmin=0.8, vmax=1, cmap='BuGn_r')
                plt.subplot(3, col, 2 * col +k+2)
                plt.title('diff')
                plt.imshow(diff, vmin=-30, vmax=30)
            # end for k
            plt.suptitle('%s-%d/%d' % (patient, j, item_num))
            plt.tight_layout()
            plt.show()
            plt.close()
        # end for items
    # end for patients




if __name__ == '__main__':
    main()
