import os
import numpy as np
import numpy.ma as ma
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    data_root_path_base = r'F:\3-Data\1-SynthRADTask2'
    data_root_path_gen = r'F:\ProjectData\6-sCT3\SynthRADTask2\out'
    patient = '2PA047'   # 2PA047  -- 020.npy  032.npy
    item = '032.npy'
    box = [90, 140, 90, 140]  # hs, he, ws, we
    series_name_list = ['DDIM_sCBCT1_2step', 'DDIM_sCBCT1_5step0', 'DDIM_sCBCT1_10step', 'DDIM_sCBCT1_200step',]
                        # 'FlowMatch_sCBCT1_15step', 'FlowMatch_sCBCT1_20step', 'FlowMatch_sCBCT1_50step']

    min_val = -1024
    max_val = 1024

    display_min_val = -160
    display_max_val = 240

    # --------
    series_num = len(series_name_list)
    # ----------------------------------
    series_path = os.path.join(data_root_path_gen, series_name_list[0])
    patient_path_gen = os.path.join(series_path, patient)
    patient_path_base = os.path.join(data_root_path_base, patient)
    patient_ct_path = os.path.join(patient_path_base, 'Full Dose Images')
    ct_path = os.path.join(patient_ct_path, item)
    base_item_path = os.path.join(patient_path_gen, item)
    if not os.path.exists(ct_path):
        raise Exception('路径不存在：%s' % ct_path)

    ct = np.load(ct_path)
    ct = np.clip(ct, min_val, max_val)

    # 绘图
    plt.figure(figsize=(10, 8))
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+00+00")
    col = series_num
    ax = plt.subplot(2, col, 1)
    plt.title('ct')
    plt.imshow(ct, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    y, x = box[0], box[2]
    height, width = box[1] - box[0], box[3] - box[2]
    rect = Rectangle((x, y), width, height,  # (x, y, width, height)
                     linewidth=2, edgecolor='green', facecolor='none')
    ax.add_patch(rect)
    plt.subplot(2, col, 2)
    plt.title('ct')
    ct1 = ct[box[0]:box[1], box[2]:box[3]]
    plt.imshow(ct1, cmap='gray', vmin=display_min_val, vmax=display_max_val)


    # sCT
    for k, series_name in enumerate(series_name_list):
        if k != 0:
            item_path = base_item_path.replace(series_name_list[0], series_name_list[k])
        else:
            item_path = base_item_path
        # end if

        sCT = np.load(item_path)
        sCT = np.clip(sCT, min_val, max_val)
        sCT = sCT[box[0]:box[1], box[2]:box[3]]

        plt.subplot(2, col, col+1+k)
        plt.title(series_name)  # predict
        plt.imshow(sCT, cmap='gray', vmin=display_min_val, vmax=display_max_val)

    # end for k
    plt.suptitle('%s-%s' % (patient, item))
    plt.tight_layout()
    plt.show()
    plt.close()



if __name__ == '__main__':
    main()
