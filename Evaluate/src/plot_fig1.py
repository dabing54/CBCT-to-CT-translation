import os
import numpy as np
import numpy.ma as ma
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 对比步数

def main():
    data_root_path_base = r'F:\3-Data\1-SynthRADTask2'
    data_root_path_gen = r'F:\ProjectData\6-sCT3\SynthRADTask2\out'
    patient = '2PA047'   # 2PA047  -- 020.npy  032.npy
    item = '032.npy'
    row_limit1 = 0 # 去除留白的行
    row_limit2 = 256

    box = [90, 140, 90, 140]  # hs, he, ws, we  原始值
    series_name_list1 = ['FlowMatch_sCBCT1_2step', 'FlowMatch_sCBCT1_10step', 'FlowMatch_sCBCT1_50step',
                         'FlowMatch_sCBCT1_100step', 'FlowMatch_sCBCT1_200step']
    series_name_list2 = ['DDIM_sCBCT1_2step', 'DDIM_sCBCT1_10step', 'DDIM_sCBCT1_50step',
                         'DDIM_sCBCT1_100step', 'DDIM_sCBCT1_200step']
    series_name_unet = 'Unet_sCBCT1'

    min_val = -1024
    max_val = 1024

    display_min_val = -160
    display_max_val = 240

    box[0] = box[0] - row_limit1
    box[1] =  box[1] - row_limit1
    y, x = box[0], box[2]
    height, width = box[1] - box[0], box[3] - box[2]

    # --------
    series_num = len(series_name_list1)
    # ----------------------------------
    series_path = os.path.join(data_root_path_gen, series_name_list1[0])
    patient_path_gen = os.path.join(series_path, patient)
    patient_path_base = os.path.join(data_root_path_base, patient)
    patient_ct_path = os.path.join(patient_path_base, 'Full Dose Images')
    ct_path = os.path.join(patient_ct_path, item)
    patient_cbct_path = os.path.join(patient_path_base, 'sCBCT1')
    cbct_path = os.path.join(patient_cbct_path, item)
    base_item_path = os.path.join(patient_path_gen, item)
    if not os.path.exists(ct_path):
        raise Exception('路径不存在：%s' % ct_path)

    ct = np.load(ct_path)
    ct = ct[row_limit1: row_limit2, :]
    ct = np.clip(ct, min_val, max_val)

    cbct = np.load(cbct_path)
    cbct = cbct[row_limit1: row_limit2, :]
    cbct = np.clip(cbct, min_val, max_val)

    # 绘图
    plt.figure(figsize=(12, 8))
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+00+00")
    col = series_num
    # cbct
    ax = plt.subplot(5, col, 1)
    # plt.title('CBCT')
    plt.imshow(cbct, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')
    rect = Rectangle((x, y), width, height,  # (x, y, width, height)
                     linewidth=2, edgecolor='green', facecolor='none')
    ax.add_patch(rect)

    # ct
    ax = plt.subplot(5, col, 2)
    # plt.title('CT')
    plt.imshow(ct, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')
    rect = Rectangle((x, y), width, height,  # (x, y, width, height)
                     linewidth=2, edgecolor='green', facecolor='none')
    ax.add_patch(rect)

    plt.subplot(5, col, 3)
    # plt.title('CT zoom')
    ct1 = ct[box[0]:box[1], box[2]:box[3]]
    plt.imshow(ct1, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')

    # unet
    item_path = base_item_path.replace(series_name_list1[0], series_name_unet)
    sCT = np.load(item_path)
    sCT = sCT[row_limit1: row_limit2, :]
    sCT = np.clip(sCT, min_val, max_val)
    ax = plt.subplot(5, col, 4)
    # plt.title('Unet')
    plt.imshow(sCT, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')
    rect = Rectangle((x, y), width, height,  # (x, y, width, height)
                     linewidth=2, edgecolor='green', facecolor='none')
    ax.add_patch(rect)
    # zoom
    sCT = sCT[box[0]:box[1], box[2]:box[3]]
    plt.subplot(5, col, 5)
    # plt.title('Unet zoom')
    plt.imshow(sCT, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')

    # sCT series1
    for k, series_name in enumerate(series_name_list1):
        if k != 0:
            item_path = base_item_path.replace(series_name_list1[0], series_name_list1[k])
        else:
            item_path = base_item_path
        # end if

        sCT = np.load(item_path)
        sCT = sCT[row_limit1: row_limit2, :]
        sCT = np.clip(sCT, min_val, max_val)

        ax = plt.subplot(5, col, col+1+k)
        # plt.title(series_name)  # predict
        plt.imshow(sCT, cmap='gray', vmin=display_min_val, vmax=display_max_val)
        plt.axis('off')
        rect = Rectangle((x, y), width, height,  # (x, y, width, height)
                         linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        # zoom
        sCT = sCT[box[0]:box[1], box[2]:box[3]]
        plt.subplot(5, col, 2*col + 1 + k)
        # plt.title(series_name)  # predict
        plt.imshow(sCT, cmap='gray', vmin=display_min_val, vmax=display_max_val)
        plt.axis('off')
    # end for k

    # sCT series2
    for k, series_name in enumerate(series_name_list2):
        if k != 0:
            item_path = base_item_path.replace(series_name_list1[0], series_name_list2[k])
        else:
            item_path = base_item_path
        # end if

        sCT = np.load(item_path)
        sCT = sCT[row_limit1: row_limit2, :]
        sCT = np.clip(sCT, min_val, max_val)

        ax = plt.subplot(5, col, 3*col+1+k)
        # plt.title(series_name)  # predict
        plt.imshow(sCT, cmap='gray', vmin=display_min_val, vmax=display_max_val)
        plt.axis('off')
        rect = Rectangle((x, y), width, height,  # (x, y, width, height)
                         linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)
        # zoom
        sCT = sCT[box[0]:box[1], box[2]:box[3]]
        plt.subplot(5, col, 4*col + 1 + k)
        # plt.title(series_name)  # predict
        plt.imshow(sCT, cmap='gray', vmin=display_min_val, vmax=display_max_val)
        plt.axis('off')
    # end for k


    plt.suptitle('%s-%s' % (patient, item))
    # plt.tight_layout()
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.1, wspace=0)
    plt.show()
    plt.close()



if __name__ == '__main__':
    main()
