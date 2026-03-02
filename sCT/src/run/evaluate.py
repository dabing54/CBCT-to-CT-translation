import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import numpy.ma as ma


class Evaluator:  # 2D、3D数据均可
    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val  # rescale的参数
        self.max_val = max_val
        self.range = max_val - min_val

    def cal_metrics(self, predict, label, mask=None, rescale=True):
        """输入为np"""
        self.valid_data(predict, label)
        out = dict()
        mask = mask == 0 if mask is not None else None  # ma数组会去掉mask中值为1的数据
        if rescale:
            assert self.min_val is not None
            assert self.max_val is not None
            predict = self.rescale(predict)
            label = self.rescale(label)
        out['SSIM'] = self.cal_ssim(predict, label, mask, data_range=self.range)  # ssim不能使用masked array，否则数组中会产生大于1的值
        if mask is not None:
            predict = ma.masked_array(predict, mask)
            label = ma.masked_array(label, mask)
        out['MAE'] = self.cal_mae(predict, label)
        out['MSE'] = self.cal_mse(predict, label)
        out['ME'] = self.cal_me(predict, label)
        out['PSNR'] = self.cal_psnr(predict, label, data_range=self.range)
        trans2, trans3 = self.trans_value(-150), self.trans_value(100)
        trans1, trans4 = self.trans_value(-1024), self.trans_value(1024)
        out['limit_MAE1'], out['limit_ME1'] = self.cal_limit_mae_me(predict, label, trans1, trans2)
        out['limit_MAE2'], out['limit_ME2'] = self.cal_limit_mae_me(predict, label, trans2, trans3)
        out['limit_MAE3'], out['limit_ME3'] = self.cal_limit_mae_me(predict, label, trans3, trans4)

        return out

    @staticmethod
    def valid_data(predict, label, mask=None):
        """验证数据有效性, predict与label均为np"""
        assert isinstance(predict, np.ndarray)
        assert isinstance(label, np.ndarray)
        assert predict.ndim in [2, 3]  # 3D 或 2D
        assert predict.shape == label.shape
        if mask is not None:
            assert isinstance(mask, np.ndarray)
            assert predict.shape == mask.shape
            assert np.max(mask) <= 1

    @staticmethod
    def cal_mae(predict, label):
        out = np.abs(predict - label)
        out = np.mean(out)
        return out

    @staticmethod
    def cal_me(predict, label):
        out = predict - label
        out = np.mean(out)
        return out

    def cal_limit_mae_me(self, predict, label, min_limit, max_limit):
        """计算指定范围内的mae, me，如软组织"""
        limit_mask = np.bitwise_or(label > max_limit, label < min_limit)
        new_mask = np.bitwise_or(label.mask, limit_mask) if isinstance(label, np.ma.MaskedArray) else limit_mask
        predict = np.ma.copy(predict)
        label = np.ma.copy(label)
        predict.mask = new_mask
        label.mask = new_mask
        mae = self.cal_mae(predict, label)
        me = self.cal_me(predict, label)
        return mae, me

    @staticmethod
    def cal_mse(predict, label):
        out = predict - label
        out = np.power(out, 2).mean()
        return out

    @staticmethod
    def cal_ssim(predict, label, mask=None, data_range=None):
        # 计算ssim前应保持数据范围为[0,max]
        assert predict.min() >= 0
        assert label.min() >= 0
        # ssim的win_size默认为7
        _, ssim = structural_similarity(predict, label, channel_axis=None, data_range=data_range, full=True)
        ssim = ma.masked_array(ssim, mask) if mask is not None else ssim
        ssim = ssim.mean()
        return ssim

    def cal_psnr(self, predict, label, data_range):
        # psnr = peak_signal_noise_ratio(label, predict, data_range=1)  # 这个包的函数无法只计算一个区域,即使使用ma数组也不行
        mse = self.cal_mse(predict, label)
        psnr = 10 * np.log10((data_range ** 2) / mse)
        return psnr

    def rescale(self, img):
        """将输入数据clip到[-1, 1]，再rescale到[min, max], 最后再减去min,使得数据位于[0, max-min]"""
        img = np.clip(img, -1, 1)
        img = (img + 1) / 2 * (self.max_val - self.min_val) + self.min_val
        img = np.around(img).astype(np.float32)  # 后续指标计算都是float,用int若差异过大会超出数据范围
        img = np.clip(img, self.min_val, self.max_val)
        img = img - self.min_val  # 计算ssim时需要
        return img

    def trans_value(self, old):
        new = old - self.min_val  # 因为图像是将值转换为【0，max-min】的
        return new


