import os
import numpy as np

from cbct_simulate import CBCTSimulator
from plot import plot_two_3d_img


def main():
    data_root_path = r'F:\3-Data\1-SynthRADTask2'

    angle_range = 220  # 360
    n_row, n_col, n_slice = 256, 256, 40
    edge = 2  # 额外补充层面，防止缺信号
    # --------------------------
    patients = os.listdir(data_root_path)
    patients = [patient for patient in patients if patient.startswith('2P')]  # TODO
    # patients = patients[:1]
    simulator = CBCTSimulator(n_row, n_col, n_slice, angle_range=angle_range, edge=edge)
    patient_num = len(patients)
    for j, patient in enumerate(patients):
        print('progress:%d/%d' % (j, patient_num))
        patient_path = os.path.join(data_root_path, patient)
        ct_path = os.path.join(patient_path, 'Full Dose Images')
        ct = read_series_img_npy(ct_path)
        ct_num = ct.shape[0]
        batchs = int(np.ceil(ct_num / n_slice))
        result_list = list()
        for i in range(batchs):
            s, e = i * n_slice, (i + 1) * n_slice
            ct_sub = ct[s:e]
            act_num = ct_sub.shape[0]
            if act_num == n_slice:
                out = simulator.run(ct_sub)
                result_list.append(out)
            else:
                extend_num = n_slice - act_num
                ct_sub = expand_imgs(ct_sub, extend_num)
                out = simulator.run(ct_sub)
                out = out[:act_num]
                result_list.append(out)
            # end if
        # end for
        result = np.concatenate(result_list, axis=0)
        result = result.round().astype(np.int16)
        # save
        series_path = os.path.join(patient_path, 'sCBCT3')  # TODO
        save_3d_img_to_npy(result, series_path, idx_s=0, dtype=np.int16)
    # end for

def expand_imgs(img, extend_num):
    post = np.tile(img[-1], (extend_num, 1, 1))
    img = np.concatenate([img, post], axis=0)
    return img

def read_series_img_num(series_path):
    items = os.listdir(series_path)
    num = len(items)
    return num

def read_series_img_npy(series_path, idx_s=None, idx_e=None):
    items = os.listdir(series_path)
    assert items[0] == '000.npy'
    if idx_s is not None:
        assert idx_e is not None
        assert idx_s < idx_e
        items = items[idx_s:idx_e]
    else:  # idx_s is None
        assert idx_e is None
    # 读取
    result = list()
    for i, item in enumerate(items):
        item_path = os.path.join(series_path, item)
        data = np.load(item_path)
        result.append(data)
    # end for
    result = np.stack(result, axis=0)
    return result

def save_2d_img_to_npy(img, patient_path, series_name, idx, dtype):
    assert isinstance(patient_path, str)
    assert isinstance(img, np.ndarray)
    series_path = os.path.join(patient_path, series_name)
    if not os.path.exists(series_path):
        os.makedirs(series_path)
    img_name = str(idx).rjust(3, '0') + '.npy'
    item_path = os.path.join(series_path, img_name)
    img = img.round().astype(dtype)
    np.save(item_path, img)


def save_3d_img_to_npy(img, series_path, idx_s, dtype):
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3
    img = img.astype(dtype)
    if not os.path.exists(series_path):
        os.makedirs(series_path)
    nums = img.shape[0]
    for i in range(nums):
        idx = idx_s + i
        item_name = str(idx).rjust(3, '0') + '.npy'
        item_path = os.path.join(series_path, item_name)
        np.save(item_path, img[i])


if __name__ == '__main__':
    main()
