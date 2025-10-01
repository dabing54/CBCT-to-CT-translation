import os

import pandas as pd


def main():
    csv_root_path = r'D:\华为云盘\L1发展\进展项目\1-sCT\3-结果\1-提取'
    save_root_path = r'D:\华为云盘\L1发展\进展项目\1-sCT\3-结果\2-分组'
    obj_col_name = 'psnr'
    # 'mae', 'rmse', 'me', 'ssim', 'ms_ssim', 'psnr', 'vifp', 'ncc', 'laplacian_var'
    # 'grad_mag', 'high_freq_energy_ratio', 'gray_var', 'brenner', 'entropy'

    #
    save_path = os.path.join(save_root_path, obj_col_name + '.csv')
    items = os.listdir(csv_root_path)
    items.sort(key=len)  # 排序
    df = pd.DataFrame()
    for i, item in enumerate(items):
        item_name = item.split('.')[0]
        item_path = os.path.join(csv_root_path, item)
        dfi = pd.read_csv(item_path)
        if i == 0:
            df[obj_col_name] = dfi.iloc[:, 0]  # 取第1列
        # end if
        df[item_name] = dfi[obj_col_name]
    # end for
    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    main()
