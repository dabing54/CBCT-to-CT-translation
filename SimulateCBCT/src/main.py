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
    patients = [patient for patient in patients if patient.startswith('2PA')]
    patients = patients[:1]
    for patient in patients:
        patient_path = os.path.join(data_root_path, patient)
        ct_path = os.path.join(patient_path, 'Full Dose Images')
        ct = read_series_img_npy(ct_path, idx_s=50, idx_e=n_slice+50)
        simulator = CBCTSimulator(n_row, n_col, n_slice, angle_range=angle_range, edge=edge)
        out = simulator.run(ct)
        plot_two_3d_img(ct, out, 'CT', 'sCBCT', v_min=-200, v_max = 200)
    # end for


def read_series_img_num(series_path):
    items = os.listdir(series_path)
    num = len(items)
    return num

def read_series_img_npy(series_path, idx_s=None, idx_e=None):
    items = os.listdir(series_path)
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


if __name__ == '__main__':
    main()
