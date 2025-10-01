import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    data_root_path_base = r'F:\3-Data\1-SynthRADTask2'
    data_root_path_gen = r'F:\ProjectData\6-sCT3\SynthRADTask2\out'
    out_path = r'E:\DaBing54\Desktop\out'
    patient = '2PA047'  # 2PA047  -- 020.npy  032.npy
    item = '020.npy'
    series_name_list = ['FlowMatch_sCBCT1_50step', 'DDIM_sCBCT1_50step', 'Unet_sCBCT1']

    dpi = 600

    box = [87, 112, 97, 122]  # hs, he, ws, we  原始值
    box2 = [122, 147, 140, 165]

    row_limit1 = 65  # 去除留白的行
    row_limit2 = 205

    box[0] = box[0] - row_limit1
    box[1] = box[1] - row_limit1
    y, x = box[0], box[2]
    height, width = box[1] - box[0], box[3] - box[2]

    box2[0] = box2[0] - row_limit1
    box2[1] = box2[1] - row_limit1
    y2, x2 = box2[0], box2[2]
    height2, width2 = box2[1] - box2[0], box2[3] - box2[2]

    min_val = -1024
    max_val = 1024

    display_min_val = -160
    display_max_val = 240

    series_path = os.path.join(data_root_path_gen, series_name_list[0])
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

    mask = np.load(mask_path)
    mask = mask[row_limit1: row_limit2, :]

    ct = np.load(ct_path)
    ct = ct[row_limit1: row_limit2, :]
    ct = np.clip(ct, min_val, max_val)

    cbct = np.load(cbct_path)
    cbct = cbct[row_limit1: row_limit2, :]
    cbct = np.clip(cbct, min_val, max_val)

    sct_list = list()
    for series_name in series_name_list:
        item_path = base_item_path.replace(series_name_list[0], series_name)
        sct = np.load(item_path)
        sct = sct[row_limit1: row_limit2, :]
        sct = np.clip(sct, min_val, max_val)
        sct[mask == 0] = min_val
        sct_list.append(sct)
    # end for

    # body外置0
    ct[mask == 0] = min_val
    cbct[mask == 0] = min_val

    col = len(sct_list) + 2
    # 1
    plt.figure(figsize=(5, 5))
    plt.imshow(cbct, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    ax = plt.gca()
    rect = Rectangle((x, y), width, height,  # (x, y, width, height)
                     linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    rect2 = Rectangle((x2, y2), width2, height2,  # (x, y, width, height)
                     linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect2)
    plt.axis('off')
    save_path = os.path.join(out_path, 'cbct.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    # sub1
    sub_img = cbct[box[0]:box[1], box[2]:box[3]]
    plt.figure(figsize=(5, 5))
    plt.imshow(sub_img, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')
    save_path = os.path.join(out_path, 'cbct_sub1.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    # sub2
    sub_img = cbct[box2[0]:box2[1], box2[2]:box2[3]]
    plt.figure(figsize=(5, 5))
    plt.imshow(sub_img, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')
    save_path = os.path.join(out_path, 'cbct_sub2.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()


    # 2.
    plt.figure(figsize=(5, 5))
    plt.imshow(ct, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    ax = plt.gca()
    rect = Rectangle((x, y), width, height,  # (x, y, width, height)
                     linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    rect2 = Rectangle((x2, y2), width2, height2,  # (x, y, width, height)
                      linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect2)
    plt.axis('off')
    save_path = os.path.join(out_path, 'ct.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    #sub1
    sub_img = ct[box[0]:box[1], box[2]:box[3]]
    plt.figure(figsize=(5, 5))
    plt.imshow(sub_img, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')
    save_path = os.path.join(out_path, 'ct_sub1.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    # sub2
    sub_img = ct[box2[0]:box2[1], box2[2]:box2[3]]
    plt.figure(figsize=(5, 5))
    plt.imshow(sub_img, cmap='gray', vmin=display_min_val, vmax=display_max_val)
    plt.axis('off')
    save_path = os.path.join(out_path, 'ct_sub2.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()


    # 3.
    for i, sct in enumerate(sct_list):
        plt.figure(figsize=(5, 5))
        plt.imshow(sct, cmap='gray', vmin=display_min_val, vmax=display_max_val)
        ax = plt.gca()
        rect = Rectangle((x, y), width, height,  # (x, y, width, height)
                         linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        rect2 = Rectangle((x2, y2), width2, height2,  # (x, y, width, height)
                          linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect2)
        plt.axis('off')
        save_path = os.path.join(out_path, series_name_list[i] + '.png')
        plt.savefig(save_path, dpi=dpi)
        plt.close()
        # sub1
        sub_img = sct[box[0]:box[1], box[2]:box[3]]
        plt.figure(figsize=(5, 5))
        plt.imshow(sub_img, cmap='gray', vmin=display_min_val, vmax=display_max_val)
        plt.axis('off')
        save_path = os.path.join(out_path, series_name_list[i] + '_sub1.png')
        plt.savefig(save_path, dpi=dpi)
        plt.close()
        # sub2
        sub_img = sct[box2[0]:box2[1], box2[2]:box2[3]]
        plt.figure(figsize=(5, 5))
        plt.imshow(sub_img, cmap='gray', vmin=display_min_val, vmax=display_max_val)
        plt.axis('off')
        save_path = os.path.join(out_path, series_name_list[i] +  '_sub2.png')
        plt.savefig(save_path, dpi=dpi)
        plt.close()


    # end for


if __name__ == '__main__':
    main()
