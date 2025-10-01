import os

from torch.utils.data import DataLoader

from dataset.transform import TrainTransform, TestTransform
from dataset.dataset import BaseData


class DataLoaderBuilder:
    def __init__(self, dataset_name, data_root_path, patient_list_path, img_size, transform_cfg, pin_memory, num_workers,
                 ldct_group_list=None):
        self.dataset_name = dataset_name
        self.data_root_path = data_root_path
        self.patient_list_path = patient_list_path
        self.img_size = img_size
        self.transform_cfg = transform_cfg
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.pad_val = transform_cfg.min_val
        self.apply_augment = transform_cfg.apply_augment
        self.ldct_group_list = ldct_group_list

    def create_data_loader(self, group, batch_size, idx=None, is_shuffle=None):
        dataset = self.create_dataset(group, idx)
        if idx is not None:
            patient =  dataset[1]
            dataset = dataset[0]
        if is_shuffle is None:  # 若未指定shuffle,则自动推断
            is_shuffle = True if group in ['train'] else False
        data_loader = DataLoader(dataset, batch_size, shuffle=is_shuffle, num_workers=self.num_workers,
                                 pin_memory=self.pin_memory)
        if idx is not None:
            return data_loader, patient
        else:
            return data_loader

    def create_dataset(self, group, idx=None):
        patient_list_path = os.path.join(self.patient_list_path, group + '.txt')
        patients = self.read_patients(patient_list_path)
        if idx is not None:  # 只取第idx个患者
            patients = [patients[idx]]
        print('use dataset group: %s, num=%d' % (group, len(patients)))
        is_train = group == 'train'
        if is_train and self.apply_augment:
            transform = TrainTransform(self.img_size, self.transform_cfg)
        else:
            transform = TestTransform(self.img_size, self.transform_cfg)
        # end if
        base_series_list = ['Full Dose Images', 'Low Dose Images', 'Mask']  # 定义文件组织结构
        if self.dataset_name == 'EdgeNeck':
            base_series_list = ['ct', 'cbct', 'mask']
        # end if
        ldct_series_list = [base_series_list[1] + ' ' + item if item.startswith('P') else item
                            for item in self.ldct_group_list]
        if group in ['valid', 'test']:
            ldct_series_list = ldct_series_list[:1]  # 验证、测试时只取第一个
        del base_series_list[1]

        dataset = BaseData(self.data_root_path, patients, self.img_size, self.pad_val, base_series_list,
                           ldct_series_list, transform)
        if idx is not None:
            return dataset, patients[0]
        else:
            return dataset

    def get_patients_num(self, group):
        patient_list_path = os.path.join(self.patient_list_path, group + '.txt')
        patients = self.read_patients(patient_list_path)
        return len(patients)

    @staticmethod
    def read_patients(patient_list_path):
        with open(patient_list_path, 'r') as f:
            patients = f.readlines()
        patients = [patient.strip() for patient in patients]
        patients = [patient for patient in patients if not patient.startswith('#')]
        for i in range(len(patients)):
            patient = patients[i]
            patient_info = patient.split('\t')
            assert len(patient_info) in [1, 3]
            patient_info = tuple(patient_info)
            patients[i] = patient_info
        return patients
