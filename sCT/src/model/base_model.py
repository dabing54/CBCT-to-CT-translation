import os
import time
import warnings
import torch
import numpy as np
from abc import ABC, abstractmethod

from run.lr_scheduler import lr_scheduler_multi_step


class BaseModel(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.model_name
        self.load_model_file = cfg.load_model_file
        self.model_path = cfg.model_path  # 过程中添加的属性
        self.device = cfg.device  # 过程中添加的属性
        self.ldct_group_list = cfg.data.ldct_group_list
        # status
        self.is_train = cfg.is_train
        self.is_continue_train = cfg.is_continue_train
        self.is_test = cfg.is_test
        # model, optim
        self.net = None
        self.optimizer = None
        self.lr_scheduler = None
        # record
        self.log_point = 0
        self.loss_item_names = ['loss']  # 用于外部标识记录

    def after_model_inited(self):
        """模型初始化后的操作"""
        if self.is_train:
            self.init_model_auxiliary()
        # 加载模型
        if self.is_continue_train or self.is_test:
            self.load_model()
        # 验证model路径是否有遗留.pth文件
        if self.is_train:
            self.verify_model_path()
        self.to_device()

    def init_model_auxiliary(self):
        # 定义优化器
        lr = self.cfg.optim.lr
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        # 定义lr scheduler
        warmup_num = self.cfg.lr_scheduler.warmup
        milestone_list = self.cfg.lr_scheduler.milestone_list
        gamma = self.cfg.lr_scheduler.gamma
        self.lr_scheduler = lr_scheduler_multi_step(self.optimizer, warmup_num, milestone_list, gamma)

    def verify_model_path(self):
        """验证model路径是否有遗留.pth文件"""
        items = os.listdir(self.model_path)
        items = [item for item in items if item.endswith('.pth')]
        if self.is_continue_train:
            items = [int(item.split('_')[-1].split('.')[0]) for item in items]
            items = np.array(items)
            if np.max(items) > self.log_point:
                raise Exception('check point路径中存在大于当前继续训练起点的log point的.pth文件')
            # end if
        else:  # 初始训练
            if len(items) > 0:
                raise Exception('check point路径中有遗留.pth文件，请先删除')
            # end if
        # end if

    def load_model(self):
        """加载模型"""
        model_path = os.path.join(self.model_path, self.load_model_file)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.log_point = checkpoint['log_point']
        self.net.load_state_dict(checkpoint['model'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            # 将optimizer参数放到device上
            # 需要以下这一段, 不要的话继续训练时好像会报错
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
                # end for1
            # end for2
        # end if
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # end func

    def save_model(self):
        """保存模型"""
        checkpoint = {
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'log_point': self.log_point,
        }
        ldct_group_name = ''.join(self.ldct_group_list)
        model_name = self.model_name + '_' + ldct_group_name + '_logpoint_' + str(self.log_point) + '.pth'
        model_path = os.path.join(self.model_path, str(model_name))
        if os.path.exists(model_path):  # 覆盖存储警告
            warnings.warn('文件夹中存在同样目标路径名的模型，发生覆盖存储操作！')
        # end if
        torch.save(checkpoint, model_path)
        # end func

    def get_current_lr(self):
        """获取当前学习率"""
        return self.optimizer.state_dict()['param_groups'][0]['lr']
        # end func

    def update_lr(self):
        """更新学习率"""
        self.lr_scheduler.step()
        # end func

    def set_train(self):
        """设置为训练模式"""
        self.net.train()

    def set_eval(self):
        """设置为评估模式"""
        self.net.eval()

    def zero_grad(self):
        """清空梯度"""
        self.optimizer.zero_grad()

    def step(self):
        """更新参数"""
        self.optimizer.step()

    def to_device(self):
        self.net.to(self.device)

    def get_loss_item_names(self):
        return self.loss_item_names

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def cal_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def optim_param(self, *args, **kwargs):
        pass


