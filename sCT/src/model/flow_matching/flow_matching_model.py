import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchcfm import ExactOptimalTransportConditionalFlowMatcher
from torchcfm import SchrodingerBridgeConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModelWrapper
from torchdyn.core import NeuralODE

from model.base_model import BaseModel


class FlowMatchModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.out_mask_weight = cfg.train.out_mask_weight

        self.n_step = cfg.flow_match.n_step

        self.flow_match = None
        self.node = None

        self.init()


    def init(self):
        self.init_model()
        self.after_model_inited()
        self.init_flow_match()
        self.init_node()

    def init_model(self):
        # 读取UNet配置
        img_d = self.cfg.net.img_size
        img_c = self.cfg.net.img_c
        out_c = self.cfg.net.out_c
        n_channels = self.cfg.net.n_channels
        channel_mult = self.cfg.net.channel_mult
        num_blocks = self.cfg.net.num_blocks
        attention_res = self.cfg.net.attention_res
        num_heads = self.cfg.net.num_heads
        dropout = self.cfg.net.dropout
        use_scale_shift_norm = self.cfg.net.use_scale_shift_norm

        dim = (img_c, img_d, img_d)

        assert out_c == img_c

        self.net = UNetModelWrapper(dim=dim, num_res_blocks=num_blocks, num_channels=n_channels,
                                    channel_mult=channel_mult, num_heads=num_heads, attention_resolutions=attention_res,
                                    dropout=dropout, use_scale_shift_norm=use_scale_shift_norm)

    def init_flow_match(self):
        flow_model_name = self.cfg.flow_match.flow_match_name
        if flow_model_name == 'otcfm':
            self.flow_match = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        elif flow_model_name == 'sbcfm':
            self.flow_match = SchrodingerBridgeConditionalFlowMatcher(sigma=1.0, ot_method='exact')
        else:
            raise NotImplementedError(flow_model_name)
        # end if

    def init_node(self):
        self.node = NeuralODE(self.net, solver='euler', sensitivity='adjoint')  # euler速度快结果还好
        # self.node = NeuralODE(self.net, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

    def forward(self, *args, **kwargs):
        pass

    def predict(self, cbct):
        with torch.no_grad():
            device = cbct.device
            t_seq = torch.linspace(0, 1, self.n_step, device=device)
            traj = self.node.trajectory(cbct, t_seq)
        # end with
        traj = traj.clip(-1, 1)
        # 以下代码用于打印中间数据
        # import os
        # import matplotlib.pyplot as plt
        # save_root_path = r'E:\DaBing54\Desktop\flow'
        # os.mkdir(save_root_path)
        # n = len(traj)
        # for i in range(n):  # 已验证traj0是原始输入数据
        #     img = traj[i].cpu().numpy()[0][0]
        #     img = img * 1024
        #     plt.figure(figsize=(5, 5))
        #     plt.imshow(img, cmap='gray', vmin=-160, vmax=240)
        #     plt.axis('off')
        #     save_path = os.path.join(save_root_path, str(i) + '.png')
        #     plt.savefig(save_path, dpi=600)
        #     plt.close()
        # 以上代码用于打印中间数据

        # 以下代码用于保存中间数据
        # save_root_path = r'E:\DaBing54\Desktop\data_flow100'
        # imgs = traj.cpu().numpy()[:, 0, 0]
        # imgs = np.round(imgs * 1024).astype(np.int16)
        # n = len(imgs)
        # for i in range(n):
        #     save_path = os.path.join(save_root_path, str(i).rjust(3, '0') + '.npy')
        #     np.save(save_path, imgs[i])
        # end for
        # 以上代码用于保存中间数据

        out = traj[-1]  # 最后一个
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
        # 均匀采样
        t, xt, ut = self.flow_match.sample_location_and_conditional_flow(y, x0)  # 对应于论文中的x0, x1
        vt = self.net(t, xt)
        loss = self.mse_loss_with_mask(ut, vt, mask)
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
