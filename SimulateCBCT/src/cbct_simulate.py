import numpy as np
import astra
from skimage.transform import resize
from skimage import exposure
from scipy.ndimage import gaussian_filter1d, gaussian_filter


class CBCTSimulator:
    def __init__(self, n_row, n_col, n_slice, angle_range=360, u0=0.02, edge=0):
        n_slice = n_slice + edge * 2
        self.edge = edge
        self.angle_range = angle_range  # 单位°
        self.n_row, self.n_col, self.n_slice = n_row, n_col, n_slice
        self.u0 = u0
        self.proj_geom, self.proj_id = None, None
        self.vol_geom = astra.create_vol_geom(n_row, n_col, n_slice)
        self.init_proj_geo()

        # mask  FDK不支持
        # half = n_row / 2
        # c = np.linspace(-half, half, n_row)
        # x, y = np.meshgrid(c, c)
        # mask = np.array((x**2 + y**2 < half ** 2), dtype=np.bool)
        # mask = np.tile(mask, (n_slice, 1, 1))
        # self.mask_id = astra.data3d.create('-vol', self.vol_geom, mask)



    def __del__(self):
        if self.proj_id is not None:
            astra.projector.delete(self.proj_id)
        # if self.mask_id is not None:
        #     astra.data3d.delete(self.mask_id)

    @ staticmethod
    def get_default_geo_para():
        # geo_para = {  # varian
        #     'det_row': 768,
        #     'det_col': 1024,
        #     'det_space': 0.398,
        #     'sad': 1000,
        #     'aid': 500,  # 旋转中心与探测器矩阵间的距离
        #     'angle_factor': 1.8  # 角度范围与角度数的比例
        # }
        geo_para = {
            'det_row': 256,  # 384
            'det_col': 768,
            'det_space': 0.8,   # 0.8
            'sad': 1000,
            'aid': 500,  # 旋转中心与探测器矩阵间的距离
            'angle_factor': 1.8  # 角度范围与角度数的比例
        }
        return geo_para

    def init_proj_geo(self):
        """通过距离缩放，维持锥角不变，像素与mm"""
        geo_para = self.get_default_geo_para()
        det_row = geo_para['det_row']
        det_col = geo_para['det_col']
        det_space = geo_para['det_space']
        sad = geo_para['sad']
        aid = geo_para['aid']
        angle_factor = geo_para['angle_factor']
        # 创建
        angle_num = int(self.angle_range * angle_factor)
        angle_range_rad = self.angle_range / 180 * np.pi
        angles = np.linspace(0, angle_range_rad, angle_num,False)
        self.proj_geom = astra.create_proj_geom('cone', det_space, det_space, det_row, det_col, angles, sad, aid)

    def check_img_info(self, img):
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[0] == self.n_slice
        assert img.shape[1] == self.n_row
        assert img.shape[2] == self.n_col


    def trans_hu_to_u(self, img):
        img = self.u0 + img * self.u0 / 1024
        return img

    def trans_u_to_hu(self, img):
        img = (img - self.u0) / self.u0 * 1024
        img = np.clip(img, a_min=-1024, a_max=None)
        return img

    def fdk_recon(self, proj_data):
        rec_id = astra.data3d.create('-vol', self.vol_geom)
        proj_id = astra.data3d.create('-sino', self.proj_geom, proj_data)
        cfg = astra.astra_dict('FDK_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        cfg['FilterType'] = 'hamming'  # 'hann'  'hamming'
        cfg['option'] = dict()
        # cfg['option']['ReconstructionMaskId'] = self.mask_id  # FDK不支持
        cfg['option']['ShortScan'] = True if self.angle_range < 360 else False  # TODO 可以提升图像质量
        # cfg['option']['FilterCutOff'] = 0.8 # 滤波截止频率  Hamming使用

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        result = astra.data3d.get(rec_id)
        # 释放资源
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(proj_id)
        return result


    def cal_sino(self, img):
        proj_id, proj_data = astra.create_sino3d_gpu(img, self.proj_geom, self.vol_geom)
        astra.data3d.delete(proj_id)
        return proj_data


    def run(self, img):
        img = img.astype(np.float32)
        if self.edge != 0:
            img = self.expland_img(img, self.edge)
        self.check_img_info(img)
        img = self.trans_hu_to_u(img)
        sino = self.cal_sino(img)
        sino = self.add_sino_noise(sino)
        img = self.fdk_recon(sino)
        img = self.trans_u_to_hu(img)
        if self.edge != 0:
            img = self.unexpland_img(img, self.edge)
        return img

    @staticmethod
    def expland_img(img, edge):
        pre = np.tile(img[0], (edge, 1, 1))
        post = np.tile(img[-1], (edge, 1, 1))
        img = np.concatenate([pre, img, post], axis=0)
        return img

    @staticmethod
    def unexpland_img(img, edge):
        img = img[edge:-edge]
        return img

    @staticmethod
    def add_sino_noise(sino, I0=12e4, scatter_fraction=0.01,beam_hard=1.01, sigma=5):
        sino = np.asarray(sino, dtype=np.float32)
        # 泊松噪声   # I = I0 * exp(-p)
        np.exp(-sino, out=sino)
        sino *= I0
        sino = np.random.poisson(sino).astype(np.float32)
        np.clip(sino, 1, I0, out=sino)  # 防止对数计算中的零值
        sino /= I0
        np.log(sino, out=sino)
        np.negative(sino, out=sino)  # 取负

        # 散射模拟 sino : row, angle, col
        scatter = gaussian_filter1d(sino, sigma=sigma, axis=0)  # 这样节省内存，且运算快
        scatter = gaussian_filter1d(scatter, sigma=sigma, axis=2)
        scatter *= scatter_fraction
        sino += scatter

        # 模拟线束硬化
        np.power(sino, beam_hard, out=sino)  # 轻微的线束硬化效应
        return sino





