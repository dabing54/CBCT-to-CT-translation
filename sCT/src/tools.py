import os
import numpy as np


def create_dir(father_path, dir_name=None):
    """在指定目录中创建文件夹，并返回创建的文件夹路径"""
    obj_path = os.path.join(father_path, dir_name) if dir_name is not None else father_path
    if not os.path.exists(obj_path):
        os.makedirs(obj_path)
    return obj_path


def format_dict(d):
    """格式化字典，创建新字典，不覆盖"""
    formatted_dict = dict()
    for k, v in d.items():
        # 1. 去除np包装
        if isinstance(v, np.floating):
            v = float(v)
        elif isinstance(v, np.integer):
            v = int(v)
        # end if
        # 2. 四舍五入
        if isinstance(v, float):
            v = round(v, 2)
        # end if
        formatted_dict[k] = v
    return formatted_dict


def extract_dict(d, k_list):
    """从dict中提取部分键生成新的dict"""
    extracted_dict = dict()
    for k in k_list:
        extracted_dict[k] = d[k]
    return extracted_dict


def group_extract_dict(d, g_list):
    """根据键分组，从字典中提取目标键，生成新的字典列表"""
    out = list()
    for g in g_list:
        extracted_dict = extract_dict(d, g)
        out.append(extracted_dict)
    return out
