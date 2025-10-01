import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from run.run_base import RunBase
from run.fix_evaluate import FixEvaluator


class Trainer(RunBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.batch_num = cfg.train.batch_num

        self.init_data_loader()
        self.fix_evaluator = FixEvaluator(cfg, self.data_loader_builder, self.model)

    def init_data_loader(self):
        """创建data loader"""
        batch_size = self.cfg.train.batch_size
        test_batch = self.cfg.test.test_batch
        self.train_loader = self.data_loader_builder.create_data_loader(group='train', batch_size=batch_size)
        self.valid_loader = self.data_loader_builder.create_data_loader(group='valid', batch_size=test_batch)

    def para_check(self):
        if len(self.train_loader) < self.batch_num:
            warnings.warn('batch_num值大于了train loader长度')
        # end

    def run(self):
        print('开始训练...')
        last_save_log_point = -1
        for epoch in range(self.epochs):
            self.model.log_point += 1
            print('epoch: %d/%d -- check point: %d' % (epoch + 1, self.epochs, self.model.log_point))
            self.do_train()
            if self.model.log_point % self.save_gap == 0:
                self.model.save_model()
                last_save_log_point = self.model.log_point
            if self.model.log_point % self.valid_gap == 0:
                self.do_valid()
            if self.model.log_point % self.sample_gap == 0:
                self.do_fixed_evaluate()
            # end if
        # end for
        if last_save_log_point != self.model.log_point:  # 以免发生覆盖存储
            self.model.save_model()
        # 训练完成
        time_spend = time.time() - self.time_star
        print("Time spend: %d min %d s" % (time_spend // 60, time_spend % 60))
        print('训练完成')

    # end func

    def do_train(self):
        self.model.set_train()
        loss_np = np.zeros(self.batch_num)
        loop = tqdm(enumerate(self.train_loader), total=self.batch_num, leave=True, desc='Train', file=sys.stdout)
        for i, (ct, cbct, mask) in loop:
            # 判断结束
            if i >= self.batch_num:
                break
            # end if
            ct = ct.to(self.device)
            cbct = cbct.to(self.device)

            loss = self.model.optim_param(cbct, ct, mask)  # input, label, mask
            # 打印输出
            loop.set_postfix(loss=loss)
            loss_np[i] = loss
        # end for
        loss_np = loss_np.mean(axis=0)
        if self.model.log_point >= self.log_point_record_start:
            self.train_writer.add_scalar('lr', self.model.get_current_lr(), self.model.log_point)
            self.train_writer.add_scalar('loss', loss_np, self.model.log_point)
            # end for
        # end if
        self.model.update_lr()

    def do_valid(self):
        self.model.set_eval()
        loss_np = np.zeros(len(self.valid_loader))
        loop = tqdm(enumerate(self.valid_loader), total=len(self.valid_loader), leave=True, desc='Valid:', file=sys.stdout)
        with torch.no_grad():
            for i, (ct, cbct, mask) in loop:
                ct = ct.to(self.device)
                cbct = cbct.to(self.device)

                loss = self.model.cal_loss(cbct, ct, mask)  # input, label, mask
                # 打印输出
                loop.set_postfix(loss=loss)
                loss_np[i] = loss
            # end for
        # end with
        loss_np = loss_np.mean(axis=0)
        if self.model.log_point >= self.log_point_record_start:
            self.valid_writer.add_scalar('loss', loss_np, self.model.log_point)
        # end if

    def do_fixed_evaluate(self):
        log_point = self.model.log_point
        title = 'log_point_' + str(log_point).rjust(4, '0') + '.png'
        save_path = os.path.join(self.sample_path, title)
        metrics = self.fix_evaluator.run(save_plot_path=save_path)
        metrics_item = ['MAE', 'ME', 'PSNR', 'SSIM',]
                        # 'limit_MAE1', 'limit_MAE2', 'limit_MAE3',
                        # 'limit_ME1', 'limit_ME2', 'limit_ME3']
        for item in metrics_item:
            self.valid_writer.add_scalar(item, metrics[item], log_point)
