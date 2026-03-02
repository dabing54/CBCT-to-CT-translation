import torch

from model.base_model import BaseModel
from model.DDIM.unet import UNet
from model.DDIM.spaced_diffusion import SpacedDiffusion


class DDIMModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.out_mask_weight = cfg.train.out_mask_weight

        self.n_step = cfg.ddpm.n_step
        self.n_step_used = cfg.ddpm.n_step_used
        self.beta_schedule = cfg.ddpm.beta_schedule
        self.noise_d = cfg.ddpm.noise_d
        self.loss_start_res = cfg.ddpm.loss_start_res
        self.loss_type = cfg.ddpm.loss_type
        self.xt_from_y = cfg.ddpm.xt_from_y
        self.clip_denoised = cfg.ddpm.clip_denoised
        t_sampler_name = cfg.ddpm.t_sampler_name
        assert t_sampler_name == 'uniform'  # 目前只实现了均匀采样

        self.diffusion = None

        self.init()

    def init(self):
        self.init_model()
        self.after_model_inited()
        self.init_ddpm()

    def init_model(self):
        # 读取UNet配置
        img_d = self.cfg.net.img_size
        img_c = self.cfg.net.img_c
        out_c = self.cfg.net.out_c
        n_channels = self.cfg.net.n_channels
        channel_mult = self.cfg.net.channel_mult
        num_res_blocks = self.cfg.net.num_res_blocks
        attention_res = self.cfg.net.attention_res
        num_heads = self.cfg.net.num_heads
        dropout = self.cfg.net.dropout
        dropout_start_res = self.cfg.net.dropout_start_res
        use_scale_shift_norm = self.cfg.net.use_scale_shift_norm
        use_skip_connection_coef = self.cfg.net.use_skip_connection_coef
        use_first_down = self.cfg.net.use_first_down

        self.net = UNet(img_c, out_c, n_channels, channel_mult, num_res_blocks, attention_res, num_heads=num_heads,
                        dropout=dropout, dropout_start_res=dropout_start_res, use_scale_shift_norm=use_scale_shift_norm,
                        use_skip_connection_coef=use_skip_connection_coef, use_first_down=use_first_down, in_size=img_d)

    def init_ddpm(self):
        kwargs = dict()
        kwargs['device'] = self.device
        kwargs['noise_d'] = self.noise_d
        kwargs['img_d'] = self.cfg.net.img_size
        kwargs['loss_type'] = self.loss_type
        kwargs['loss_start_res'] = self.loss_start_res
        kwargs['xt_from_y'] = self.xt_from_y
        kwargs['eps_model'] = self.net
        kwargs['predict_type'] = self.cfg.ddpm.predict_type
        kwargs['ddim_eta'] = self.cfg.ddpm.ddim_eta
        self.diffusion = SpacedDiffusion(self.n_step, self.n_step_used, self.beta_schedule, **kwargs)

    def forward(self, *args, **kwargs):
        pass

    def predict(self, cbct):
        return self.diffusion.p_sample_loop(cbct, clip_denoised=self.clip_denoised)

    def _cal_loss(self, y, x0, mask):
        # 均匀采样
        b = x0.shape[0]
        t = torch.randint(0, self.n_step, (b,)).to(self.device)
        loss = self.diffusion.cal_loss(y, t, x0, mask, out_mask_weight=self.out_mask_weight)
        loss = loss.mean()
        return loss

    def cal_loss(self, y, x, mask):
        loss = self._cal_loss(y, x, mask)
        loss = loss.item()
        return loss

    def optim_param(self, y, x, mask):
        self.zero_grad()
        loss = self._cal_loss(y, x, mask)
        loss.backward()
        self.step()
        loss = loss.item()
        return loss
