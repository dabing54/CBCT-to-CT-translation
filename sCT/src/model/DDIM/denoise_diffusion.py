import os.path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
"""
根据improve DDPM开源代码更改， 基本忠实于原代码
根据DDIM进行修改
"""


class GaussianDiffusion:
    def __init__(self, betas, used_step, n_stpes, **kwargs):
        """
        * eps_model: nn.Module
        * betas: np数组, float 64
        """
        self.eps_model = kwargs['eps_model']
        self.n_steps = n_stpes
        self.t_seq = used_step  # 真实的t, 很多函数中的t实际上是索引
        self.device = kwargs['device']
        self.loss_type = kwargs['loss_type']
        self.xt_from_y = kwargs['xt_from_y']
        self.loss_start_res = kwargs['loss_start_res']
        self.predict_type = kwargs['predict_type']
        ddim_eta = kwargs['ddim_eta']
        # 以下为运算所需要的变量
        assert len(betas.shape) == 1
        assert (betas > 0).all() and (betas <= 1).all()
        assert int(betas.shape[0]) == self.n_steps
        assert betas.dtype == np.float64  # 中间过程有些值很小，为了提升精度
        self.beta = betas
        self.alpha = 1.0 - self.beta
        self.alpha_bar = np.cumprod(self.alpha, axis=0)  # 数值递减  alpha_bar_t
        self.alpha_bar_prev = np.append(1.0, self.alpha_bar[:-1])  # alpha_bar_{t-1}
        self.post_mean_coef1 = betas * np.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        self.post_mean_coef2 = (1.0 - self.alpha_bar_prev) * np.sqrt(self.alpha) / (1.0 - self.alpha_bar)
        self.sqrt_alpha_bar = np.sqrt(self.alpha_bar)
        self.sqrt_alpha_bar_prev = np.sqrt(self.alpha_bar_prev)
        self.sqrt_one_minus_alpha_bar = np.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha_bar = np.sqrt(1.0 / self.alpha_bar)
        self.sqrt_recipm1_alpha_bar = np.sqrt(1.0 / self.alpha_bar - 1.0)
        # 方差
        sigma2 = (1-self.alpha_bar_prev) / (1-self.alpha_bar) * (1 - self.alpha_bar/self.alpha_bar_prev)
        self.sigma = ddim_eta * np.sqrt(sigma2)


    def extract_to_tensor(self, arr, t_steps, reshape=True):
        """取与时间步t对应的数据，reshape"""
        c = np.array(arr[t_steps.cpu()])
        c = np.expand_dims(c, axis=0) if c.ndim == 0 else c  # 当t_steps只有一个值的时候，从np中取出的c则为标量
        c = torch.from_numpy(c).float().to(self.device)  # 将数据类型转换为float32
        c = c.reshape(-1, 1, 1, 1) if reshape else c
        return c


    def q_sample(self, x0, t, noise=None):
        """q(x_t | x_0)"""
        noise = torch.randn_like(x0) if noise is None else noise
        assert noise.shape == x0.shape
        mean = self.extract_to_tensor(self.sqrt_alpha_bar, t) * x0
        std = self.extract_to_tensor(self.sqrt_one_minus_alpha_bar, t)
        out = mean + std * noise
        return out


    def p_mean_var(self, xt, t, y, clip_denoised=False):
        """p(x_{t-1} | x_t)，同时预测x0
        model_out: 若非空，则在该函数中不用调用self.eps_model()函数，直接使用传入的值
        """
        b, c = xt.shape[:2]
        assert t.shape[0] == b
        real_t = self.extract_to_tensor(self.t_seq, t, reshape=False)
        print(real_t)  # todo 待删除
        model_out = self.eps_model(xt, real_t, y)
        # 方差
        model_sigma = self.extract_to_tensor(self.sigma, t)
        # predict x0
        if self.predict_type == 'epsilon':
            pred_eps = model_out
            pred_x0 = self.predict_x0_from_eps(xt, t, model_out)
        elif self.predict_type == 'v_predict':
            pred_x0, pred_eps = self.predict_x0_from_v(xt, t, model_out)
        else:
            raise NotImplementedError(self.predict_type)
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1, 1)
            # pred_eps = pred_eps.clamp(-1, 1)  # 不需要
        # 平均
        coef1 = self.extract_to_tensor(self.sqrt_alpha_bar_prev, t)
        mid_coef = self.extract_to_tensor(self.alpha_bar_prev, t)
        coef2 = torch.sqrt(1 - mid_coef - model_sigma ** 2 )
        model_mean = coef1 * pred_x0 + coef2 * pred_eps
        return model_mean, model_sigma

    def predict_x0_from_eps(self, xt, t, eps):
        """论文中公式9反过来运用，而不是公式13"""
        coef1 = self.extract_to_tensor(self.sqrt_recip_alpha_bar, t)
        coef2 = self.extract_to_tensor(self.sqrt_recipm1_alpha_bar, t)
        x0 = coef1 * xt - coef2 * eps
        return x0

    def predict_x0_from_v(self, xt, t, v):
        coef1 = self.extract_to_tensor(self.sqrt_alpha_bar, t)
        coef2 = self.extract_to_tensor(self.sqrt_one_minus_alpha_bar, t)
        x0 = coef1 * xt - coef2 * v
        eps = coef1 * v + coef2 * xt
        return x0, eps

    def get_v_from_noise(self, x0, t, noise):
        coef1 = self.extract_to_tensor(self.sqrt_alpha_bar, t)
        coef2 = self.extract_to_tensor(self.sqrt_one_minus_alpha_bar, t)
        v = coef1 * noise - coef2 * x0
        return v

    def p_sample(self, xt, t, y, clip_denoised=False):
        """Sample: x_t -> x_{t-1}"""
        mean, sigma = self.p_mean_var(xt, t, y, clip_denoised=clip_denoised)
        noise = torch.randn_like(xt)
        nonzero_mask = (t > 1).float().view(-1, 1, 1, 1)  # no noise when t == 1
        sample = mean + nonzero_mask * sigma * noise
        return sample

    def p_sample_loop(self, y, shape=None, noise=None, clip_denoised=False):
        """Sample: x_t -> x_0"""
        if shape is None:
            shape = y.shape
        assert isinstance(shape, (tuple, list))
        x = torch.randn(*shape, device=self.device) if noise is None else noise
        if self.xt_from_y:  # 若忠实原文，此处应为false
            # x = torch.clone(y)
            t = torch.tensor([self.n_steps - 1] * shape[0], device=self.device)
            x = self.q_sample(y, t)
        ind = list(range(0, self.n_steps))[::-1]
        ind = tqdm(ind, desc='\t\tSample')
        #  --- 以下代码用于绘制中间数据
        # import matplotlib.pyplot as plt
        # import os
        # save_root_path = r'E:\DaBing54\Desktop\ddim'
        # os.mkdir(save_root_path)
        # img = x.cpu().numpy()[0, 0]
        # img = np.clip(img, -1, 1)
        # img = img * 1024
        # plt.figure(figsize=(5, 5))
        # plt.imshow(img, cmap='gray', vmin=-160, vmax=240)
        # plt.axis('off')
        # save_path = os.path.join(save_root_path, '0.png')
        # plt.savefig(save_path, dpi=600)
        # plt.close()
        # mm = 0
        # --- 以上代码用于绘制中间数据
        # --以下代码用于保存中间数据
        # save_root_path = r'E:\DaBing54\Desktop\data_ddim100'
        # img = x.cpu().numpy()[0, 0]
        # img = np.clip(img, -1, 1)
        # img = img * 1024
        # img = np.round(img).astype(np.int16)
        # save_path = os.path.join(save_root_path, '000.npy')
        # np.save(save_path, img)
        # mm = 0
        # --以上代码用于保存中间数据

        with torch.no_grad():
            for i in ind:
                t = torch.tensor([i] * shape[0], device=self.device)  # 在跨步采样中，此处t实际上为索引
                x = self.p_sample(x, t, y, clip_denoised=clip_denoised)
                # --- 以下代码用于绘制中间数据
                # img = x.cpu().numpy()[0, 0]
                # img = np.clip(img, -1, 1)
                # img = img * 1024
                # plt.figure(figsize=(5, 5))
                # plt.imshow(img, cmap='gray', vmin=-160, vmax=240)
                # plt.axis('off')
                # mm += 1  # i为倒叙，不能直接用i
                # save_path = os.path.join(save_root_path,  str(mm) + '.png')
                # plt.savefig(save_path, dpi=600)
                # plt.close()
                # --- 以上代码用于绘制中间数据
                # -- 以下代码用于保存中间数据
                # img = x.cpu().numpy()[0, 0]
                # img = np.clip(img, -1, 1)
                # img = img * 1024
                # img = np.round(img).astype(np.int16)
                # mm += 1
                # save_path = os.path.join(save_root_path, str(mm).rjust(3, '0') + '.npy')
                # np.save(save_path, img)
                # print(self.t_seq)
                # --以上代码用于保存中间数据
            # end for
        # end with
        return x

    def mse_loss(self, pred, label, weight=None, flag='l2'):
        # 实现了MSEloss和MAEloss
        assert flag in ['l1', 'l2']
        if flag == 'l2':
            out = (pred - label) ** 2
        else:
            out = torch.abs(pred - label)
        if weight is not None:
            out = out * weight
        out = self.mean_flat_with_mask(out, weight)
        return out

    def mse_loss_mult_res(self, pred, label, weight=None, flag='l2'):
        start_res = self.loss_start_res
        ori_res = pred.shape[-1]  # 假定h与w一样
        n = int(np.floor(np.log2(ori_res / start_res))) + 1
        coef_mult = np.arange(0, n, 1)
        coef_mult = np.power(2, coef_mult)
        coef_mult = coef_mult / coef_mult[n // 2]
        loss_mult = list()
        for i in range(n):
            loss = self.mse_loss(pred, label, weight, flag)
            loss_mult.append(loss)
            if i + 1 < n:
                pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
                label = F.avg_pool2d(label, kernel_size=2, stride=2)
                weight = F.avg_pool2d(weight, kernel_size=2, stride=2) if weight is not None else None
            # end if
        # end for
        loss = 0
        for i in range(n):
            loss  += loss_mult[i] * coef_mult[i]
        # end for
        return loss

    @staticmethod
    def mean_flat_with_mask(arr, mask=None):
        """返回平均值, arr: b, ....,   out: b"""
        obj_dim = list(range(1, len(arr.shape)))
        if mask is None:
            out = torch.mean(arr, dim=obj_dim)
        else:
            out = torch.sum(arr, obj_dim)
            n = torch.count_nonzero(mask, obj_dim) + 1
            out = out / n
        return out  # 除batch维度外,其余均flat

    @staticmethod
    def get_weight(label, mask=None, out_mask_weight=0, is_balance_weight=False):
        if is_balance_weight:
            edge = torch.linspace(-1, 1, 41)  # 分40个区间赋予权重
            # fixed_val = torch.tensor([0.925, 1.400, 1.378, 0.659, 1.021, 1.663, 2.071, 2.231,
            #                           2.208, 2.042, 1.861, 1.679, 1.489, 1.316, 1.147, 0.982,
            #                           0.745, 0.114, 0.043, 0.046, 0.017, 0.041, 0.263, 0.352,
            #                           0.391, 0.436, 0.490, 0.557, 0.631, 0.716, 0.804, 0.893,
            #                           0.983, 1.085, 1.190, 1.297, 1.415, 1.544, 1.681, 0.194])
            # fixed_val = torch.tensor([1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
            #                           1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
            #                           1.00, 2.18, 0.83, 0.88, 0.33, 0.78, 1.00, 1.00,
            #                           1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
            #                           1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00])
            # fixed_val = torch.tensor([3, 3, 3, 3, 3, 3, 3, 3,
            #                           3, 3, 3, 3, 3, 3, 3, 3,
            #                           1.5, 1, 1, 0.5, 1, 1.5, 2, 2,
            #                           2, 2, 2, 2, 2, 2, 2, 2,
            #                           2, 2, 2, 2, 2, 2, 2, 1.5])
            fixed_val = torch.tensor([10, 10, 10, 10, 10, 10, 10, 10,
                                      10, 10, 10, 10, 10, 10, 10, 10,
                                      2, 1.1, 1.1, 1, 1.1, 1.5, 2, 2,
                                      2, 2, 2, 2, 2, 2, 2, 2,
                                      2, 2, 2, 2, 2, 2, 2, 1.5])
            ind = torch.searchsorted(edge, label, right=False) - 1
            ind = torch.clamp(ind, 0)
            weight = fixed_val[ind]
        else:
            weight = torch.ones_like(label)
        if mask is not None:
            weight[mask == 0] = out_mask_weight  # body外的值
        return weight

    @staticmethod
    def get_weight2(x0, y, mask=None, out_mask_weight=0):
        dis = np.abs((x0 - y).numpy())
        dis[mask == 0] = 0
        point = np.percentile(dis, 95)
        weight = torch.ones_like(x0)
        weight[dis > point] = 0
        if mask is not None:
            weight[mask == 0] = out_mask_weight  # body外的值
        return weight

    def cal_loss(self, y, t, x0, mask=None, noise=None, out_mask_weight=0, is_balance_weight=False):
        noise = torch.randn_like(x0) if noise is None else noise
        if self.predict_type == 'epsilon':
            label = noise
        elif self.predict_type == 'v_predict':
            v = self.get_v_from_noise(x0, t, noise)
            label = v
        else:
            raise NotImplementedError(self.predict_type)
        # end if
        xt = self.q_sample(x0, t, noise)
        real_t = self.extract_to_tensor(self.t_seq, t, reshape=False)  # 仅用于最终输入unet
        weight = self.get_weight(x0.cpu(), mask, out_mask_weight, is_balance_weight).to(self.device)
        # weight = self.get_weight2(x0.cpu(), y.cpu(), mask, out_mask_weight).to(self.device)
        assert self.loss_type in ['l1', 'l2']
        model_out = self.eps_model(xt, real_t, y)
        # mse部分
        loss = self.mse_loss_mult_res(model_out, label, weight, self.loss_type)
        return loss
