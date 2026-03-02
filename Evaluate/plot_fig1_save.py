import os
import numpy as np
import numpy.ma as ma
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from cal_metrics_2d import center_crop

# 对比步数

def main():
    data_root_path_base = r'F:\3-Data\2-sdfy_cbct_to_ct\pelvic'
    data_root_path_gen = r'F:\ProjectData\6-sCT4\out'
    out_path = r'E:\DaBing54\Desktop\out'
    patient = '252930'   # 2PA047  -- 020.npy  032.npy
    item = '021.npy'
    row_limit1 = 0 # 去除留白的行
    row_limit2 = 256
    
    dpi = 600
    box_color = 'red'

    box = [60, 110, 105, 155]  # hs, he, ws, we  原始值
    # box = [30, 80, 92, 142]
    series_name_list1 = ['Flow-2step', 'Flow-10step', 'Flow-50step',
                         'Flow-100step', 'Flow-200step']
    series_name_list2 = ['DDIM-2step', 'DDIM-10step', 'DDIM-50step',
                         'DDIM-100step', 'DDIM-200step']
    series_name_unet = 'Unet'

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
    patient_ct_path = os.path.join(patient_path_base, 'CT')
    ct_path = os.path.join(patient_ct_path, item)
    patient_cbct_path = os.path.join(patient_path_base, 'sCBCT2')
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
    plt.figure(figsize=(5, 5))
    plt.imshow(cbct, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')
    rect = Rectangle((x, y), width, height,  # (x, y, width, height)
                     linewidth=2, edgecolor=box_color, facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
    save_path = os.path.join(out_path, 'cbct.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()

    # ct
    plt.imshow(ct, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')
    rect = Rectangle((x, y), width, height,  # (x, y, width, height)
                     linewidth=2, edgecolor=box_color, facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
    save_path = os.path.join(out_path, 'ct.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()

    ct1 = ct[box[0]:box[1], box[2]:box[3]]
    plt.imshow(ct1, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')
    save_path = os.path.join(out_path, 'ct_zoom.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()

    # unet
    item_path = base_item_path.replace(series_name_list1[0], series_name_unet)
    sCT = np.load(item_path)
    sCT = center_crop(sCT, ct)
    sCT = sCT[row_limit1: row_limit2, :]
    sCT = np.clip(sCT, min_val, max_val)
    plt.imshow(sCT, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')
    rect = Rectangle((x, y), width, height,  # (x, y, width, height)
                     linewidth=2, edgecolor=box_color, facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
    save_path = os.path.join(out_path, 'unet.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    # zoom
    sCT = sCT[box[0]:box[1], box[2]:box[3]]
    plt.imshow(sCT, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')
    save_path = os.path.join(out_path, 'unet_zoom.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()

    # sCT series1
    for k, series_name in enumerate(series_name_list1):
        if k != 0:
            item_path = base_item_path.replace(series_name_list1[0], series_name_list1[k])
        else:
            item_path = base_item_path
        # end if

        sCT = np.load(item_path)
        sCT = center_crop(sCT, ct)
        sCT = sCT[row_limit1: row_limit2, :]
        sCT = np.clip(sCT, min_val, max_val)

        # plt.title(series_name)  # predict
        plt.imshow(sCT, cmap='gray', vmin=display_min_val, vmax=display_max_val)
        plt.axis('off')
        rect = Rectangle((x, y), width, height,  # (x, y, width, height)
                         linewidth=2, edgecolor=box_color, facecolor='none')
        ax = plt.gca()
        ax.add_patch(rect)
        save_path = os.path.join(out_path, series_name + '.png')
        plt.savefig(save_path, dpi=dpi)
        plt.close()
        # zoom
        sCT = sCT[box[0]:box[1], box[2]:box[3]]
        # plt.title(series_name)  # predict
        plt.imshow(sCT, cmap='gray', vmin=display_min_val, vmax=display_max_val)
        plt.axis('off')
        save_path = os.path.join(out_path, series_name + '_zoom.png')
        plt.savefig(save_path, dpi=dpi)
        plt.close()
    # end for k

    # sCT series2
    for k, series_name in enumerate(series_name_list2):
        if k != 0:
            item_path = base_item_path.replace(series_name_list1[0], series_name_list2[k])
        else:
            item_path = base_item_path
        # end if

        sCT = np.load(item_path)
        sCT = center_crop(sCT, ct)
        sCT = sCT[row_limit1: row_limit2, :]
        sCT = np.clip(sCT, min_val, max_val)

        # plt.title(series_name)  # predict
        plt.imshow(sCT, cmap='gray', vmin=display_min_val, vmax=display_max_val)
        plt.axis('off')
        rect = Rectangle((x, y), width, height,  # (x, y, width, height)
                         linewidth=2, edgecolor=box_color, facecolor='none')
        ax = plt.gca()
        ax.add_patch(rect)
        save_path = os.path.join(out_path, series_name + '.png')
        plt.savefig(save_path, dpi=dpi)
        plt.close()
        # zoom
        sCT = sCT[box[0]:box[1], box[2]:box[3]]
        # plt.title(series_name)  # predict
        plt.imshow(sCT, cmap='gray', vmin=display_min_val, vmax=display_max_val)
        plt.axis('off')
        save_path = os.path.join(out_path, series_name + '_zoom.png')
        plt.savefig(save_path, dpi=dpi)
        plt.close()
    # end for k




if __name__ == '__main__':
    main()
