import numpy as np
import torch

from model.base_model import BaseModel
from model.Unet.unet import UNet


class UnetModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.out_mask_weight = cfg.train.out_mask_weight

        self.net = None

        self.init()


    def init(self):
        self.init_model()
        self.after_model_inited()

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


    def forward(self, *args, **kwargs):
        pass

    def predict(self, cbct):
        with torch.no_grad():
            out = self.net(cbct)
        # end with
        return out

    def mse_loss_with_mask(self, predict, label, mask):
        loss = (predict - label) ** 2
        device = loss.device
        if mask is not None:
            assert self.out_mask_weight is not None
            weight = torch.ones_like(mask)
            weight[mask==0] = self.out_mask_weight
            weight = weight.to(device)
            loss = loss * weight
            # avg
            obj_dim = list(range(1, len(mask.shape)))
            n = torch.count_nonzero(mask, obj_dim) + 1
            n = n.to(device)
            loss = torch.sum(loss, obj_dim)
            loss = loss / n
            loss = torch.mean(loss)
        else:
            loss = torch.mean(loss)
        return loss


    def _cal_loss(self, y, x0, mask):
        sct = self.net(y)
        loss = self.mse_loss_with_mask(x0, sct, mask)
        return loss

    def cal_loss(self, y, x, mask):
        loss = self._cal_loss(y, x, mask)
        return loss.item()

    def optim_param(self, y, x, mask):
        self.zero_grad()
        loss = self._cal_loss(y, x, mask)
        loss.backward()
        self.step()
        return loss.item()
