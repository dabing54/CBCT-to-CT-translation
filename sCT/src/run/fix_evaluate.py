import numpy as np
import matplotlib.pyplot as plt
import torch

from run.evaluate import Evaluator
from tools import format_dict, group_extract_dict


class FixEvaluator:
    def __init__(self, cfg, dataloader_builder, model=None):
        self.model = model
        # fixed para
        self.fixed_group = cfg.fix_pred.fixed_group
        self.fixed_patient_idx = cfg.fix_pred.fixed_patient_idx
        self.fixed_slice_idx = cfg.fix_pred.fixed_slice_idx
        # display
        self.display_min = cfg.display.min_val
        self.display_max = cfg.display.max_val
        self.is_remove_out_mask_to_display = cfg.display.is_remove_out_mask_to_display
        # data
        self.data_loader_builder = dataloader_builder
        self.min_val = cfg.transform.min_val
        self.max_val = cfg.transform.max_val
        # evaluator
        self.evaluator = Evaluator(min_val=self.min_val, max_val=self.max_val)
        self.metrics = None
        self.metrics_o = None  # 原始指标
        self.use_mask_eval = cfg.test.use_mask_eval
        # fix data, tensor对象
        self.fixed_x = None  # ct
        self.fixed_y = None  # cbct
        self.predict_x = None  # sct
        self.fixed_mask = None
        # init
        self.init_fixed_data()

    def init_fixed_data(self):
        data_loader, _ = self.data_loader_builder.create_data_loader(self.fixed_group, batch_size=1, is_shuffle=False,
                                                                  idx=self.fixed_patient_idx)
        data_loader = iter(data_loader)
        x, y, mask = None, None, None
        for i in range(self.fixed_slice_idx):
            x, y, mask = next(data_loader)
        self.fixed_y = y
        self.fixed_x = x
        self.fixed_mask = mask if self.use_mask_eval else None

    def run(self, show_plot=False, save_plot_path=False):
        assert self.model is not None
        device = self.model.device
        self.model.set_eval()
        fixed_y = self.fixed_y.to(device)

        sct = self.model.predict(fixed_y)

        result = self.set_predict_x(sct.cpu())
        if show_plot:
            self.show_plot()
        if save_plot_path:
            self.save_plot(save_plot_path)
        return result

    def set_predict_x(self, x):
        """x: tensor对象，b, c, h, w"""
        self.predict_x = x
        self.cal_metrics()
        return self.metrics

    def cal_metrics(self):
        assert self.predict_x is not None
        assert self.fixed_x is not None
        predict, label, mask, ori = self.get_np_from_tensor(self.predict_x, self.fixed_x, self.fixed_mask, self.fixed_y)
        metrics = self.evaluator.cal_metrics(predict, label, mask, rescale=True)
        self.metrics = metrics
        metrics_o = self.evaluator.cal_metrics(ori, label, mask, rescale=True)
        self.metrics_o = metrics_o

    @staticmethod
    def get_np_from_tensor(*args):
        out = list()
        for data in args:
            data = data.numpy()[0, 0] if data is not None else None  # b, c, h, w -> h, w
            out.append(data)
        # end for
        return out

    @staticmethod
    def remove_out_mask(img: np.ndarray, mask: np.ndarray, val):
        """输入np类型"""
        img[mask == 0] = val
        return img

    def save_plot(self, save_path):
        """绘图，不显示，保持到文件"""
        self.base_plot()
        plt.savefig(save_path)
        plt.close()

    def show_plot(self):
        """绘图，并显示"""
        self.base_plot()
        plt.show()
        plt.close()

    def rescale_clip_for_display(self, img):
        """输入[-1,1] -> [min, max] -> [display_min, display_max]"""
        # img np类型
        img = self.rescale_clip_img(img)
        # display clip
        img = np.clip(img, self.display_min, self.display_max)
        return img

    def rescale_clip_img(self, img):
        """输入[-1,1] -> [min, max]"""
        # img np类型
        img = np.clip(img, -1, 1)
        img = (img + 1) / 2 * (self.max_val - self.min_val) + self.min_val
        img = np.around(img).astype(int)
        img = np.clip(img, self.min_val, self.max_val)
        return img


    def base_plot(self):
        plt.figure(figsize=(15, 15))
        plt.rcParams['font.family'] = 'Times New Roman'
        mngr = plt.get_current_fig_manager()
        mngr.window.wm_geometry("+0+100")
        assert self.predict_x is not None
        assert self.fixed_x is not None
        assert self.fixed_y is not None

        temp = self.get_np_from_tensor(self.predict_x, self.fixed_x, self.fixed_y, self.fixed_mask)
        predict_x, fixed_x, fixed_y, mask = temp

        fixed_y_display = self.rescale_clip_for_display(fixed_y)
        fixed_x_display = self.rescale_clip_for_display(fixed_x)
        predict_x_display = self.rescale_clip_for_display(predict_x)

        if mask is not None and self.is_remove_out_mask_to_display:
            fixed_x_display = self.remove_out_mask(fixed_x_display, mask, self.display_min)
            fixed_y_display = self.remove_out_mask(fixed_y_display, mask, self.display_min)
            predict_x_display = self.remove_out_mask(predict_x_display, mask, self.display_min)

        # 1.
        plt.subplot2grid((3, 3), (0, 0))
        plt.imshow(fixed_y_display, cmap='gray')
        # plt.axis('off')
        plt.title('CBCT')
        # 2.
        plt.subplot2grid((3, 3), (0, 2))
        plt.imshow(predict_x_display, cmap='gray')
        # plt.axis('off')
        plt.title('sCT')
        # 3.
        plt.subplot2grid((3, 3), (0, 1))
        plt.imshow(fixed_x_display, cmap='gray')
        # plt.axis('off')
        plt.title('CT')
        # 4.
        plt.subplot2grid((3, 3), (1, 0))
        error = predict_x_display - fixed_x_display
        plt.imshow(error, cmap='seismic', vmin=-300, vmax=300)
        # plt.axis('off')
        plt.colorbar()
        plt.title('sCT - CT')
        # 5.
        plt.subplot2grid((3, 3), (1, 1), rowspan=1, colspan=2)
        if mask is not None:
            points_x = fixed_x[mask == 1]
            points_y = predict_x[mask == 1]
        else:
            points_x = fixed_x.flatten()
            points_y = predict_x.flatten()
        plt.plot([-1, 1], [-1, 1], color='darkgreen', linestyle='--')
        plt.scatter(points_x, points_y, s=1)
        plt.title('scatter: CT -- sCT')
        # 6.
        plt.subplot2grid((3, 3), (2, 0))
        error = fixed_y_display - fixed_x_display
        plt.imshow(error, cmap='seismic', vmin=-300, vmax=300)
        plt.axis('off')
        # plt.colorbar()
        plt.title('CBCT - CT')
        # 7.
        plt.subplot2grid((3, 3), (2, 1), rowspan=1, colspan=2)
        if mask is not None:
            points_x = fixed_x[mask == 1]
            points_y = fixed_y[mask == 1]
        else:
            points_x = fixed_x.flatten()
            points_y = fixed_y.flatten()
        plt.plot([-1, 1], [-1, 1], color='darkgreen', linestyle='--')
        plt.scatter(points_x, points_y, s=1)
        plt.title('scatter: CT -- CBCT')

        title = 'Group:%s-Patient:%s-Slice:%s' % (self.fixed_group, self.fixed_patient_idx, self.fixed_slice_idx)
        metrics_group = [['MAE', 'ME', 'SSIM', 'PSNR']]
                         # ['limit_MAE1', 'limit_MAE2', 'limit_MAE3'],
                         # ['limit_ME1', 'limit_ME2', 'limit_ME3']]
        metrics_dict = format_dict(self.metrics)
        dict_list = group_extract_dict(metrics_dict, metrics_group)
        for d in dict_list:
            title = title + '\nsCT: ' + str(d)

        metrics_dict = format_dict(self.metrics_o)
        dict_list = group_extract_dict(metrics_dict, metrics_group)
        for d in dict_list:
            title = title + '\nCBCT: ' + str(d)

        plt.suptitle(title)

