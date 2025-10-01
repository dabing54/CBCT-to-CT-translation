import os
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class BaseData(Dataset):
    """2D方式加载"""
    def __init__(self, data_root_path, patients, img_size, pad_val, base_series_list, ldct_series_list, transform=None):
        self.data_root_path = data_root_path
        self.patients = patients  # 格式[(p_id, ...),(),...]
        self.img_size = img_size
        self.pad_val = pad_val
        self.base_series_list = base_series_list
        self.ldct_series_list = ldct_series_list
        self.ldct_series_num = len(ldct_series_list)
        self.img1_path_list = list()  # 高质量CT
        self.img2_path_list = list()  # 低质量CT
        self.mask_path_list = list()
        self.init_img_path(border_num=0)  # 上下界各丢弃0层

        self.transform = transform

    def __len__(self):
        return len(self.img1_path_list)

    def __getitem__(self, idx):
        img1_path = self.img1_path_list[idx]
        img2_path = self.img2_path_list[idx]
        mask_path = self.mask_path_list[idx]
        # print(img1_path)
        # 按概率选不同的ldct
        if self.ldct_series_num > 1:
            p = np.random.uniform(0, self.ldct_series_num)
            series_idx = int(p)
            if series_idx != 0:
                img2_path = img2_path.replace(self.ldct_series_list[0], self.ldct_series_list[series_idx])
        # # 使用nib读取nii时，默认读取为float64。经测试，使用与存储类型一致的格式读取速度更快
        # img1 = nib.load(img1_path).get_fdata(dtype=np.float32).transpose((1, 0))  # h, w
        # img2 = nib.load(img2_path).get_fdata(dtype=np.float32).transpose((1, 0))
        # mask = nib.load(mask_path).get_fdata(dtype=np.float32).astype(np.bool).transpose((1, 0))
        img1 = np.load(img1_path)
        img2 = np.load(img2_path)
        mask = np.load(mask_path)

        # transform
        img1, img2, mask = self.transform(img1, img2, mask)

        # import matplotlib.pyplot as plt
        # ct = np.clip(img1[0], -0.5, 0.5)
        # cbct = np.clip(img2[0], -0.5, 0.5)
        # plt.subplot(1, 2, 1)
        # plt.imshow(ct, cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(cbct, cmap='gray')
        # plt.show()

        mask = mask.unsqueeze(0)  # mask没有通道，因此需要增加一个维度
        return img1, img2, mask  # 高质量、低质量、mask

    def init_img_path(self, border_num=0):
        for patient in self.patients:
            patient_id = patient[0]
            if len(patient) == 3:
                start, end = int(patient[1]), int(patient[2]) + 1
            else:
                start, end = None, None
            patient_path = os.path.join(self.data_root_path, patient_id)
            img1_path = os.path.join(patient_path, self.base_series_list[0])  # 高质量
            mask_path = os.path.join(patient_path, self.base_series_list[1])
            img2_path = os.path.join(patient_path, self.ldct_series_list[0])  # 低质量，默认取第0个
            item_num = len(os.listdir(img1_path))
            if start is None:
                start = 0
                end = item_num
            # end if
            start = start + border_num
            end = end - border_num

            for i in range(start, end):
                item = str(i).rjust(3, '0') + '.npy'
                img1_item_path = os.path.join(img1_path, item)
                img2_item_path = os.path.join(img2_path, item)
                mask_item_path = os.path.join(mask_path, item)
                if not os.path.exists(img1_item_path):
                    raise Exception('文件不存在，请核查%s' % img1_item_path)
                if not os.path.exists(img2_item_path):
                    raise Exception('文件不存在，请核查%s' % img2_item_path)
                if not os.path.exists(mask_item_path):
                    raise Exception('文件不存在，请核查%s' % img2_item_path)
                self.img1_path_list.append(img1_item_path)
                self.img2_path_list.append(img2_item_path)
                self.mask_path_list.append(mask_item_path)
            # end for1
        # end for 2


