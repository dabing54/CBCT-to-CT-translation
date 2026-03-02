import matplotlib.pyplot as plt
import numpy as np
import astra
import SimpleITK as sitk
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter1d
import scipy.ndimage as ndi

from gas_motion import BowelGasMotionSimulator


def resample(img: sitk.Image, obj_size:np.ndarray, is_mask=False):
    """根据obj_size执行重采样"""
    space = np.array(img.GetSpacing())  # w, h, d
    size = np.array(img.GetSize())
    obj_size = np.array(obj_size)
    obj_space = space * size / obj_size
    # 重采样
    origin = img.GetOrigin()
    direction = img.GetDirection()
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing(obj_space)
    resampler.SetSize(obj_size.tolist())
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputDirection(direction)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resample_method = sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline
    default_vale = 0 if is_mask else -1000
    out_pixel_type = sitk.sitkUInt8 if is_mask else sitk.sitkInt16
    resampler.SetInterpolator(resample_method)
    resampler.SetDefaultPixelValue(default_vale)
    resampler.SetDefaultPixelValue(out_pixel_type)
    out = resampler.Execute(img)
    return out

def cal_uniform_space_size(img:sitk.Image):
    """计算均匀间距的obj_size"""
    space = np.array(img.GetSpacing())  # w, h, d
    size = np.array(img.GetSize())
    assert space[0] == space[1]
    obj_space = (space[0], space[0], space[0])
    obj_size = size * space / np.array(obj_space)
    obj_size = np.round(obj_size).astype(int)
    return obj_size



class CBCTSimulator:
    def __init__(self):
        # 几何参数
        self.d_rows = 512  # 探测器行数  #
        self.d_cols = 786  # 探测器列数  # 腹部为half-fan模式，列数该加倍，等价于间距加倍行减半
        self.d_space = 0.776 # 探测器像素尺寸 (mm) 0.388  0.392  0.776
        self.sdd = 1500
        self.sad = 1000
        self.angle_num = 300
        self.proj_angles = np.linspace(0, 2 * np.pi, self.angle_num, endpoint=False)  # 360个投影角度
        # 参数
        self.proj_group_num = 10
        self.gas_threshold = -160  # HU
        self.body_erosion = 6  # mm
        self.resp_amp_ap = 20  # mm
        self.resp_amp_lr = 0  # mm
        self.peris_amp = 0
        self.max_influence_distance = 10
        # 物理参数
        self.scatter_sigma = 5
        self.scatter_fraction = 0.02
        self.photon_flux = 1e7  # 入射光子通量
        self.beam_hard = 1e-4  # 旧的公式1.05
        self.min_proj_value = 0.01  # 最小截断，即探测器过饱和0.1
        self.max_proj_value = 12  # 截断
        # 物理效应
        self.scatter_kernel = self._generate_scatter_kernel()  # 散射核
        self.bowtie_filter_profile = self._generate_bowtie_profile()  # 领结过滤器轮廓
        # 图像元信息
        self.size = None
        self.space = None
        self.origin = None
        self.direction = None
        # 控制参数
        self.u0 = 0.020  # 120KV下水的线性衰减系数，单位mm-1  # 0.19cm-1
        self.u_air = 0.0001
        # 存储
        # self.proj_geom = None
        # self.vol_geom = None
        self.hu = None

    def _apply_unit_trans(self):
        assert self.space is not None
        space = self.space[0]  # w, h, d
        self.sdd /= space
        self.sad /= space
        self.d_space /= space

        self.u0 *= space  # TODO


    def run(self, img:sitk.Image, mask):
        # 1. CT数据处理
        ct, mask = self._convert_sitk_to_np(img, mask)
        self.hu = ct
        # 单位换算
        self._apply_unit_trans()
        ct = self._convert_hu_to_u(ct)
        # 2. 投影
        shape = ct.shape  # d, h, w
        vol_geom = astra.create_vol_geom(shape[1], shape[2], shape[0])  # row, col, slice
        # 分组投影
        moving_cts = self._simulate_gas_motion(self.hu, ct, mask)
        out = list()
        group_projection_num = self.angle_num // self.proj_group_num
        for group_idx in range(self.proj_group_num):
            current_ct = moving_cts[group_idx]
            base_idx = group_idx * group_projection_num
            group_angles = self.proj_angles[base_idx:base_idx+group_projection_num]
            proj_geom = self._get_proj_geom(group_angles)
            proj_id, projections = astra.create_sino3d_gpu(current_ct, proj_geom, vol_geom)
            astra.data3d.delete(proj_id)
            out.append(projections)
        # end for
        projections = np.concatenate(out, axis=1)  # angle在维度1

        # 3. 投影阈处理
        proj_geom = self._get_proj_geom(self.proj_angles)
        # self.vol_geom = vol_geom
        projections = self._apply_simulate_effects(projections)
        # 4. 图像重建
        recon_id = astra.data3d.create('-vol', vol_geom)
        proj_id = astra.data3d.create('-sino', proj_geom, projections)
        cfg = astra.astra_dict('FDK_CUDA')
        cfg['ReconstructionDataId'] = recon_id
        cfg['ProjectionDataId'] = proj_id
        cfg['FilterType'] = 'ram-lak'
        cfg['option'] = dict()
        cfg['option']['ShortScan'] = True
        alg_id = astra.algorithm.create(cfg)

        astra.algorithm.run(alg_id)
        recon = astra.data3d.get(recon_id)
        # 5. 释放资源
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(recon_id)
        astra.data3d.delete(proj_id)
        # 6. 图像转换
        recon = self._convert_u_to_hu(recon)
        ct = self._convert_back_to_sitk(recon)
        return ct


    def _apply_simulate_effects(self, projections):
        # 射线硬化效应
        projections = self._apply_beam_hardening(projections)
        # 散射效应
        projections = self._apply_scatter_effect(projections)
        # 量子效应
        projections = self._apply_photon_effect(projections)
        # 探测器截断
        projections = self._apply_detector_trunc(projections)
        return projections

    def _apply_photon_effect(self, projections):
        # 泊松噪声模拟光子统计
        photon_flux = self.photon_flux * self.bowtie_filter_profile[np.newaxis, np.newaxis, :]
        photon_counts = photon_flux * np.exp(-projections)
        photon_counts_noise = np.random.poisson(photon_counts)
        # photon_counts = np.clip(photon_counts, a_min=1, a_max=photon_flux)
        ratio = np.clip(photon_counts_noise / photon_flux, a_min=1e-8, a_max=1)
        projections = -np.log(ratio)

        # 光子饥饿：空气区域相对值变小
        # mask = projections < 0.2
        # projections[mask] += np.random.normal(0, 0.02, projections.shape)[mask]
        # projections = np.clip(projections, a_min=0, a_max=np.inf)

        return projections

    def _apply_scatter_effect(self, projections):
        # 低频散射
        scatter = gaussian_filter1d(projections, sigma=self.scatter_sigma, axis=0)
        scatter = gaussian_filter1d(scatter, sigma=self.scatter_sigma, axis=2)
        scatter *= self.scatter_fraction

        projections = projections + scatter
        return projections

    def _apply_beam_hardening(self, projections):
        """应用射线硬化效"""
        # projections = np.power(projections, self.beam_hard)
        projections = projections + self.beam_hard * np.power(projections, 3)
        return projections

    def _apply_detector_trunc(self, projections):
        projections = np.clip(projections, a_min=self.min_proj_value, a_max=self.max_proj_value)

        return projections

    def _generate_bowtie_profile(self):
        """生成 Varian Edge OBI 领结过滤器的列衰减因子（中心列=1，边缘衰减更大）"""
        col_cent = self.d_cols // 2
        col_pos = (np.arange(self.d_cols) - col_cent) * self.d_space  # 各列到中心距离（mm）
        # 经验模型：Varian OBI bowtie 衰减（可拟合实测数据）
        # 形式：atten = exp(-k * |x|^2)，k 由能量/滤过决定
        k = 0.00002  # 125 kVp 典型值（需按实测校准）0.00008
        profile = np.exp(-k * np.abs(col_pos) ** 2)
        # 归一化：中心列衰减=1
        profile /= profile[col_cent]
        # plt.plot(profile)
        # plt.show()
        return profile

    def _generate_scatter_kernel(self):
        """生成散射核（高斯核近似）"""
        kernel_size = 15
        sigma = 3
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, kernel_size // 2] = 1
        kernel = ndimage.gaussian_filter(kernel, sigma)
        kernel = kernel / kernel.sum()
        return kernel

    def _convert_sitk_to_np(self, img:sitk.Image, mask):
        """图像预处理
        # 1. 各向异性像素间距处理：重采样
        # 2. sitk.Image -> np
        """
        # 存储图像元信息
        self.size = img.GetSize()
        self.space = img.GetSpacing()
        self.origin = img.GetOrigin()
        self.direction = img.GetDirection()
        # 重采样
        obj_size = cal_uniform_space_size(img)
        img = resample(img, obj_size)
        mask = resample(mask, obj_size, is_mask=True)
        # sitk -> np
        img = sitk.GetArrayFromImage(img)  # d, h, w
        img = img.astype(np.float32)
        img = np.clip(img, -1000, 3000)  # 空气-1000、骨骼3000
        mask = sitk.GetArrayFromImage(mask)
        return img, mask

    def _convert_back_to_sitk(self, hu:np.ndarray) -> sitk.Image:
        """重新构建sitk.Image"""
        # np -> sitk
        hu = hu.astype(np.float32)
        img = sitk.GetImageFromArray(hu)
        img.SetOrigin(self.origin)
        img.SetDirection(self.direction)
        space = [self.space[0], self.space[0], self.space[0]]
        img.SetSpacing(space)
        # 重采样
        img = resample(img, self.size)
        img = sitk.Cast(img, sitk.sitkInt16)
        return img

    def _convert_hu_to_u(self, hu:np.ndarray) -> np.ndarray:
        # u = (HU * 0.001) * u_water + u_water
        # u0 = self.u0 * self.space[0]  # mm-1  --> pix-1
        u0 = self.u0
        u_air = self.u_air
        u = (hu * 0.001) * u0 + u0
        u = np.clip(u, a_min=u_air, a_max=np.inf)  # 使用u_air防止值为0
        return u

    def _convert_u_to_hu(self, u:np.ndarray) -> np.ndarray:
        # u -> HU
        # u0 = self.u0 * self.space[0]
        u0 = self.u0
        hu = (u - u0) / u0 * 1000
        hu = np.clip(hu, -1000, 3000)
        return hu

    def _detect_gas_regions(self, ct: np.ndarray, mask:np.ndarray, threshold: float = -500) -> np.ndarray:
        """检测气体区域 (HU值低于阈值的区域)。"""
        # 假设输入是HU值，空气约为-1000 HU
        gas_mask = np.logical_and(ct < threshold, mask == 1).astype(np.float32)
        # 形态学操作去除小噪声，并平滑边界
        gas_mask = ndi.binary_closing(gas_mask, structure=np.ones((3, 3, 3))).astype(np.float32)
        gas_mask = ndi.binary_dilation(gas_mask, iterations=2).astype(np.float32)
        gas_mask = ndi.gaussian_filter(gas_mask, sigma=1.0)
        return gas_mask

    def _simulate_gas_motion(self, hu: np.ndarray, u: np.ndarray, body_mask: np.ndarray):
        """
        模拟气体在扫描过程中的随机运动。
        改进版本：支持多个独立的气体区域，每个区域有独立的运动。
        Returns:
            包含运动伪影的体积序列 (每个投影角度一个体积)
        """
        moving_ct = list()
        motion_simulator = BowelGasMotionSimulator(hu, u, body_mask,self.space[0], self.gas_threshold, self.body_erosion,
                                                   self.resp_amp_ap, self.resp_amp_lr, self.peris_amp, self.max_influence_distance)
        for group_idx in range(self.proj_group_num):
            resp_state = group_idx/self.proj_group_num
            out = motion_simulator.run(resp_state)
            # out = np.copy(ct)
            moving_ct.append(out)
        # end for
        # plt.imshow(u[50], cmap='gray')
        # plt.show()
        # plt.imshow(moving_ct[0][50], cmap='gray')
        # plt.show()
        # plt.imshow(moving_ct[1][50], cmap='gray')
        # plt.show()
        # plt.imshow(moving_ct[2][50], cmap='gray')
        # plt.show()
        # plt.imshow((u-moving_ct)[0][50])
        # plt.show()
        # plt.imshow((u - moving_ct)[1][50])
        # plt.show()
        # plt.imshow((u - moving_ct)[2][50])
        # plt.show()
        return moving_ct

    def _get_proj_geom(self, proj_angles):
        proj_geom = astra.create_proj_geom(
            'cone',
            self.d_space, self.d_space,
            self.d_rows, self.d_cols,
            proj_angles,
            self.sad,
            self.sdd - self.sad,
        )
        return proj_geom
