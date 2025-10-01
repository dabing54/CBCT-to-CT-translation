import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    data_root_path = r'E:\DaBing54\Desktop\data_flow100'
    out_path = r'E:\DaBing54\Desktop\out'
    sample_rate = 0.1
    use_mask = False
    dpi=600

    #
    if use_mask:
        mask_path = os.path.join(data_root_path, 'mask.npy')
        mask = np.load(mask_path)
    items = os.listdir(data_root_path)
    items = [item for item in items if not item.startswith('mask')]
    items = [item for item in items if item.endswith('.npy')]
    n = len(items)
    img_list = list()
    for i in range(n):
        item_path = os.path.join(data_root_path, items[i])
        img = np.load(item_path)
        img_list.append(img)
    # end for
    imgs = np.array(img_list)
    if use_mask:
        points = imgs[:, mask==1]  # k x n， k个数值n条曲线
    else:
        # points = imgs.reshape(imgs.shape[0], -1)
        points = imgs[:, imgs[-1]>-1024]
    points = points.T  # n x k
    # 采样
    idx = np.random.permutation(points.shape[0])
    points = points[idx]
    num = int(points.shape[0] * sample_rate)
    points = points[:num]
    n, k = points.shape
    print('绘制..')
    plt.figure(figsize=(5, 5))
    for i in range(n):
        print('progress: %.2f' % (i / n))
        y = points[i]
        plt.plot(y, color='cornflowerblue', alpha=0.3, linewidth=0.1)  # 0.3-0.1
        # plt.scatter(0, y[0], color='red', s=0.05, zorder=5)
        # plt.scatter(k - 1, y[-1], color='green', s=0.05, zorder=5)
    # end for
    ax = plt.gca()
    ax.set_xlim(0, 100)
    ax.set_ylim(-1024, 1024)
    from matplotlib.ticker import FuncFormatter
    def normalize(x_val, _):
        return f'{x_val/100:.1f}'
    ax.xaxis.set_major_formatter(FuncFormatter(normalize))
    save_path = os.path.join(out_path, '3.png')
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    # plt.show()

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Times New Roman'
    main()


