import math
import numpy as np

from model.DDIM.denoise_diffusion import GaussianDiffusion


def log(x, eps=1e-20):
    return np.log(np.clip(x, a_min=eps, a_max=None))

class SpacedDiffusion:
    """对DDPM进行包装，使反向过程可跨步执行，缩短sample时间"""

    def __init__(self, n_steps, n_steps_used, beta_schedule, **kwargs):
        self.n_steps = n_steps  # 原始DDPM使用的步数
        self.n_steps_used = n_steps_used  # 实际采样使用的步数
        self.base_steps = None
        self.used_steps = None  # 在跨步执行中，使用的步
        self.beta_schedule = beta_schedule
        self.base_betas = np.empty((0,))
        self.spaced_betas = None
        self.noise_d = kwargs['noise_d']
        self.img_d = kwargs['img_d']

        self.init_base_steps()
        self.init_used_steps()
        self.init_base_betas()
        self.init_spaced_betas()
        self.diffusion = GaussianDiffusion(self.spaced_betas, self.used_steps, self.n_steps_used, **kwargs)
        self.diffusion_train = GaussianDiffusion(self.base_betas, self.base_steps, self.n_steps, **kwargs)

    def init_base_steps(self):
        self.base_steps = np.arange(self.n_steps, dtype=int)

    def init_used_steps(self):
        assert self.n_steps_used <= self.n_steps
        # used_steps = np.linspace(0, self.n_steps - 1, self.n_steps_used)  # np.linspace取值前后均闭合
        used_steps = np.linspace(1, self.n_steps - 1, self.n_steps_used-1)
        if len(used_steps) == 1:  # TODO
            used_steps = [self.n_steps - 1]
        used_steps = np.append(np.array([0]), used_steps)
        used_steps = np.around(used_steps).astype(int)  # 必须取整，后面计算spaced_betas时需要整数的used_steps
        self.used_steps = used_steps

    def init_base_betas(self):
        if self.beta_schedule == 'linear':
            scale = 1000 / self.n_steps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            beta = np.linspace(beta_start, beta_end, self.n_steps, dtype=np.float64)
        elif self.beta_schedule == 'cosine':
            t = np.arange(0, self.n_steps, 1) / self.n_steps  # 介于0-1
            logsnr = self.logsnr_schedule_cosine(t, self.noise_d, self.img_d)
            beta = self.cal_beta_from_logsnr(logsnr)
        elif self.beta_schedule == 'shifted_cosine':
            t = np.arange(0, self.n_steps, 1) / self.n_steps  # 介于0-1
            logsnr = self.logsnr_schedule_cosine(t, self.noise_d, self.img_d)
            logsnr_shift = logsnr + 2 * np.log(self.noise_d / self.img_d)
            beta = self.cal_beta_from_logsnr(logsnr_shift)
        elif self.beta_schedule == 'shifted_cosine_interpolated':
            # noise_d1 = min(self.img_d // 2, self.noise_d * 2)
            # noise_d2 = self.noise_d // 2
            noise_d1 = self.img_d
            noise_d2 = self.noise_d
            t = np.arange(0, self.n_steps, 1) / self.n_steps  # 介于0-1
            logsnr = self.logsnr_schedule_cosine(t, self.noise_d, self.img_d)
            logsnr_shift1 = logsnr + 2 * np.log(noise_d1 / self.img_d)
            logsnr_shift2 = logsnr + 2 * np.log(noise_d2 / self.img_d)
            logsnr_interpolated = t * logsnr_shift1 + (1-t) * logsnr_shift2
            beta = self.cal_beta_from_logsnr(logsnr_interpolated)
        else:
            raise NotImplementedError(self.beta_schedule)
        self.base_betas = beta


    @staticmethod
    def logsnr_schedule_cosine(t, noise_d, img_d, logsnr_min=-15, logsnr_max=15):  # TODO min = -15
        # logSNR(t) = -2 * log (tan (pi * t / 2))
        # 考虑边界效应后，logsnr_t = -2 * log(tan(t_min + t * (t_max - t_min)))
        # 此处t介于0-1
        logsnr_max = logsnr_max + np.log(noise_d / img_d)
        logsnr_min = logsnr_min + np.log(noise_d / img_d)
        t_min = np.atan(np.exp(-0.5 * logsnr_max))
        t_max = np.atan(np.exp(-0.5 * logsnr_min))

        logsnr = -2 * log(np.tan((t_min + t * (t_max - t_min))))

        return logsnr

    @staticmethod
    def cal_beta_from_logsnr(logsnr):
        alpha_bar = 1.0 / (1 + np.exp(-logsnr))
        alpha = np.zeros_like(alpha_bar)
        alpha[0] = alpha_bar[0]
        alpha[1:] = alpha_bar[1:] / alpha_bar[:-1]
        beta = 1 - alpha
        return beta


    @staticmethod
    def func1(t):  # 此处的t对应于原文公式17中的t/T
        out = math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        return out

    def init_spaced_betas(self):
        """根据used_steps，生成新的betas"""
        if len(self.used_steps) == self.n_steps:
            self.spaced_betas = self.base_betas
        else:
            assert self.base_betas.dtype == np.float64
            alpha_bars = np.cumprod(1.0 - self.base_betas, axis=0)
            new_betas = list()
            last_alpha_bar = 1.0
            for i, alpha_bar in enumerate(alpha_bars):
                if i in self.used_steps:
                    new_betas.append(1.0 - alpha_bar / last_alpha_bar)
                    last_alpha_bar = alpha_bar
                # end if
            # end for
            self.spaced_betas = np.array(new_betas)
        # end if

    # end func


    def cal_loss(self, y, t, x0, mask, noise=None, out_mask_weight=None, is_balance_weight=None):
        return self.diffusion_train.cal_loss(y, t, x0, mask, noise=noise, out_mask_weight=out_mask_weight,
                                       is_balance_weight=is_balance_weight)

    def p_sample_loop(self, y, shape=None, noise=None, clip_denoised=False):
        return self.diffusion.p_sample_loop(y, shape=shape, noise=noise, clip_denoised=clip_denoised)

    def set_eval(self):
        self.diffusion_train.is_predict = True
        self.diffusion.is_predict = True

    def set_train(self):
        self.diffusion_train.is_predict = False

