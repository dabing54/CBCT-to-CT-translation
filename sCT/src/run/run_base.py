import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from dataset.data_load import DataLoaderBuilder
from tools import create_dir


class RunBase:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = cfg.data.dataset_name
        self.model_name = cfg.model_name
        self.experiment_name = cfg.experiment_name
        # status
        self.is_train = cfg.is_train
        self.is_continue_train = cfg.is_continue_train
        self.is_test = cfg.is_test
        # train para
        self.epochs = cfg.epochs
        self.save_gap = cfg.checkpoint.model_save_gap
        self.sample_gap = cfg.checkpoint.sample_gap
        self.valid_gap = cfg.train.valid_gap
        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # record
        self.time_star = None
        self.log_point_record_start = cfg.display.log_point_record_start
        # data
        self.data_loader_builder = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        # Summary writer
        self.train_writer = None
        self.valid_writer = None
        self.test_writer = None
        # path
        self.log_path = None
        self.model_path = None
        self.sample_path = None
        # model
        self.model = None

        # 初始化
        self.init()

    def init(self):
        print('初始化...')
        self.time_star = time.time()
        # 初始化
        self.init_checkpoint_path()
        self.init_data_loader_builder()
        self.init_writer()
        self.init_model()

    def init_model(self):
        # 传递变量
        self.cfg.model_path = self.model_path
        self.cfg.device = self.device
        # 定义model
        if self.model_name == 'FlowMatch':
            from model.flow_matching.flow_matching_model import FlowMatchModel
            self.model = FlowMatchModel(self.cfg)
        elif self.model_name == 'DDIM':
            from model.DDIM.ddim_model import DDIMModel
            self.model = DDIMModel(self.cfg)
        elif self.model_name == 'Unet':
            from model.Unet.unet_model import UnetModel
            self.model = UnetModel(self.cfg)
        else:
            raise NotImplementedError(self.model_name)
        # end if

    def init_checkpoint_path(self):
        check_point_path = os.path.join(self.cfg.checkpoint.checkpoint_path, str(self.dataset_name))
        check_point_path = os.path.join(check_point_path, str(self.model_name))
        check_point_path = os.path.join(check_point_path, str(self.experiment_name))
        self.log_path = os.path.join(check_point_path, 'log')
        self.model_path = os.path.join(check_point_path, 'model')
        self.sample_path = os.path.join(check_point_path, 'sample')
        # 创建路径
        create_dir(self.log_path)
        create_dir(self.sample_path)
        create_dir(self.model_path)

    def init_data_loader_builder(self):
        # 读取配置
        data_root_path = self.cfg.data.data_root_path
        patient_list_path = self.cfg.data.patient_list_path
        img_size = self.cfg.net.img_size
        trans_cfg = self.cfg.transform
        pin_memory = self.cfg.general.pin_memory
        num_workers = self.cfg.general.num_workers
        dataset_name = self.cfg.data.dataset_name
        ldct_group_list = self.cfg.data.ldct_group_list

        self.data_loader_builder = DataLoaderBuilder(dataset_name, data_root_path, patient_list_path, img_size,
                                                     trans_cfg, pin_memory, num_workers, ldct_group_list)
        # 因为test时需要对每个患者创建data loader，因此在此函数中不创建data loader
        # end

    def init_writer(self):
        """定义tensor board writer"""
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        if self.is_train:
            self.train_writer = self.create_writer(self.log_path, 'train')
            self.valid_writer = self.create_writer(self.log_path, 'valid')
        else:
            self.test_writer = self.create_writer(self.log_path, 'test')
        # end if

    @staticmethod
    def create_writer(log_path, group_name):
        group_log_path = create_dir(log_path, group_name)
        writer = SummaryWriter(group_log_path)
        return writer


