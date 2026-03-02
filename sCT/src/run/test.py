import os.path
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from run.run_base import RunBase
from run.evaluate import Evaluator
from tools import format_dict


class Tester(RunBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.test_batch = cfg.test.test_batch
        self.min_val = cfg.transform.min_val
        self.max_val = cfg.transform.max_val
        self.evaluator = Evaluator(self.min_val, self.max_val)

        self.patient = None
        self.is_save_sCT = cfg.test.is_save_sCT
        save_data_root_path = cfg.data.save_data_root_path
        self.series_path = os.path.join(save_data_root_path, 'out')
        if os.path.exists(self.series_path):
            raise Exception('已存在路径' + self.series_path)

    def run(self):
        print('开始测试...')
        patients_num = self.data_loader_builder.get_patients_num(group='test')
        dict_list = list()
        for i in range(patients_num):
            print('Progress: %d/%d' % (i, patients_num))
            self.test_loader, patient = self.data_loader_builder.create_data_loader('test', self.test_batch, idx=i)
            self.patient = patient[0]
            metrics = self.do_test()
            print(metrics)
            dict_list.append(metrics)
        # end for
        df = pd.DataFrame(dict_list)
        metrics_mean = df.mean()
        print(metrics_mean)

        # 完成
        time_spend = time.time() - self.time_star
        print("Time spend: %d min %d s" % (time_spend // 60, time_spend % 60))
        print('测试完成')

    @torch.no_grad()
    def do_test(self):
        self.model.set_eval()
        ct_list, sct_list, mask_list = list(), list(), list()
        # loop = tqdm(enumerate(self.test_loader), total=self.test_batch, leave=True, desc='Test', file=sys.stdout)
        data_len = len(self.test_loader)
        for i, (ct, cbct, mask) in enumerate(self.test_loader):  # 此处不需要torch.no_grad()
            print('sub progress: %d/%d' % (i, data_len))
            ct = ct.to(self.device)
            cbct = cbct.to(self.device)
            sct = self.model.predict(cbct)
            # 存储
            ct_list.append(ct.cpu().numpy())
            sct_list.append(sct.cpu().numpy())
            mask_list.append(mask.cpu().numpy())
        # end for
        assert ct_list[0].shape[1] == 1  # 输出为单通道图片
        ct = np.concatenate(ct_list, axis=0)[:, 0, :, :]  # b, c, h, w -> d, h, w
        sct = np.concatenate(sct_list, axis=0)[:, 0, :, :]
        mask = np.concatenate(mask_list, axis=0)[:, 0, :, :]
        # evaluate
        metrics = self.evaluator.cal_metrics(sct, ct, mask, rescale=True)
        metrics = format_dict(metrics)

        if self.is_save_sCT:
            sct = self.trans_back(sct)
            patient_path = os.path.join(self.series_path, self.patient)
            save_3d_img_to_npy(sct, patient_path, idx_s=0, dtype=np.int16)
        # end if
        return metrics

    def trans_back(self, img):
        img = np.clip(img, -1, 1)
        img = (img + 1) * (self.max_val - self.min_val) / 2 + self.min_val
        img = np.round(img)
        return img

def save_3d_img_to_npy(img, series_path, idx_s, dtype):
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3
    img = img.astype(dtype)
    if not os.path.exists(series_path):
        os.makedirs(series_path)
    nums = img.shape[0]
    for i in range(nums):
        idx = idx_s + i
        item_name = str(idx).rjust(3, '0') + '.npy'
        item_path = os.path.join(series_path, item_name)
        np.save(item_path, img[i])

