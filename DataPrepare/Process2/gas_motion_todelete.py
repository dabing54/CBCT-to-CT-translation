import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from scipy.interpolate import RegularGridInterpolator
import scipy.ndimage as ndi


class BowelGasMotionSimulator:
    """模拟腹部CT中肠道气体因呼吸、蠕动产生的位移变化"""
    def __init__(self, hu: np.ndarray, u: np.ndarray, body_mask: np.ndarray,
                 voxel_size: float = 1.5, hu_threshold=-900, body_erosion=6,
                 resp_amp_ap=3, resp_amp_lr=1, peris_amp=3, max_influence_distance=10):
        """
        初始化模拟器
        Args:
            hu: 3D HU值数组，shape=(d, h, w)
            u: 3D线性衰减系数数组，shape=(d, h, w)
            body_mask: 3D body掩码（1为body区域，0为背景），shape=(d, h, w)
            voxel_size: 体素分辨率/层厚 (mm/pix)，默认1.5mm
            erosion_mm: body内缩距离
        """
        self.hu = hu.astype(np.float32)
        self.u = u.astype(np.float32)
        self.body_mask = body_mask.astype(np.bool_)
        self.hu_threshold = hu_threshold
        self.voxel_size = voxel_size
        self.body_erosion = int(np.ceil(body_erosion / voxel_size))  # 内缩像素数
        self.resp_amp_ap = resp_amp_ap / voxel_size
        self.resp_amp_lr = resp_amp_lr / voxel_size
        self.peris_amp = peris_amp / voxel_size
        self.max_influence_distance = max_influence_distance / voxel_size  # 空气区域最大影响距离， 单位：像素  10

        # 初始处理
        self.erosion_body_mask = self._erode_body_mask()  # 内缩body mask
        self.gas_mask = self._identify_bowel_gas()  # 肠道气体掩码

    def run(self, resp_state) -> np.ndarray:
        # 取值
        gas_mask = self.gas_mask
        erosion_body_mask = self.erosion_body_mask
        u = self.u
        resp_amp_ap = self.resp_amp_ap
        resp_amp_lr = self.resp_amp_lr
        peris_amp = self.peris_amp
        max_influence_distance = self.max_influence_distance

        # 简化模拟呼吸运动，360°为一个呼吸周期.输入值介于【0-1】 -> [0,2pi】
        resp_state = resp_state * 2 * np.pi

        if not np.any(gas_mask): # 无空气直接返回
            return u.copy()
        # end if

        assert gas_mask.dtype == bool  # 下一句的~mask仅在bool类型能有效，若为uint8，0取反后为255
        # 计算每个体素到空气区域的欧式距离，正数表示非空气，负数表示空气区域内部
        distance_map = ndi.distance_transform_edt(~gas_mask) - ndi.distance_transform_edt(gas_mask)
        # 归一化距离场到[0,1]，仅在空气附近（距离<10像素）有位移，远处为0
        weight_map = np.clip(1.0 - (np.abs(distance_map) / max_influence_distance), 0.0, 1.0)
        # 只在body区域内应用权重
        weight_map = weight_map * erosion_body_mask.astype(np.float32)

        # 4. 生成呼吸运动位移场（低频、平滑）
        z, y, x = u.shape
        # 创建网格坐标，节省内存，使用float32
        z_coords = np.linspace(0, z - 1, z, dtype=np.float32)
        y_coords = np.linspace(0, y - 1, y, dtype=np.float32)
        x_coords = np.linspace(0, x - 1, x, dtype=np.float32)
        zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

        # 呼吸位移：基于正弦函数生成平滑的整体位移，模拟呼吸的周期性
        resp_disp = resp_amp_ap * np.sin(resp_state)
        resp_disp_lr = resp_amp_lr * np.sin(resp_state)
        # resp_dx = 0  # 不考虑AB方向
        resp_dx = resp_disp_lr * gas_mask
        resp_dy = resp_disp * gas_mask   # AP方向
        # resp_dz = resp_disp * gas_mask  # SI方向
        resp_dz = 0

        # 5. 生成蠕动运动位移场（高频、局部）
        # 使用高斯噪声生成随机位移，再平滑处理模拟蠕动的局部性
        peris_dz = peris_amp * np.random.randn(z, y, x).astype(np.float32)
        peris_dy = peris_amp * np.random.randn(z, y, x).astype(np.float32)
        peris_dx = peris_amp * np.random.randn(z, y, x).astype(np.float32)

        # 高斯平滑，降低噪声，模拟蠕动的连续特性
        peris_dz = ndi.gaussian_filter(peris_dz, sigma=2) * weight_map
        peris_dy = ndi.gaussian_filter(peris_dy, sigma=2) * weight_map
        peris_dx = ndi.gaussian_filter(peris_dx, sigma=2) * weight_map

        # 6. 总位移场 = 呼吸位移 + 蠕动位移
        dz = resp_dz + peris_dz
        dy = resp_dy + peris_dy
        dx = resp_dx + peris_dx
        # displacement_field = np.stack([dz, dy, dx], axis=-1)  # shape=(z,y,x,3)

        # 7. 基于位移场变换CT图像（核心步骤，保证准确性）
        # 计算变换后的坐标
        new_zz = zz + dz
        new_yy = yy + dy
        new_xx = xx + dx

        # 限制坐标在图像范围内，避免越界
        new_zz = np.clip(new_zz, 0, z - 1)
        new_yy = np.clip(new_yy, 0, y - 1)
        new_xx = np.clip(new_xx, 0, x - 1)

        # 创建插值器，使用线性插值保证准确性和效率
        interpolator = RegularGridInterpolator(
            (z_coords, y_coords, x_coords),
            u,
            method='linear',
            bounds_error=False,
            fill_value=0.0  # 背景填充0
        )

        # 重塑坐标用于插值，减少内存占用（按体素展平）
        coords = np.stack([new_zz.ravel(), new_yy.ravel(), new_xx.ravel()], axis=1)
        moved_u = interpolator(coords).reshape(z, y, x)

        return moved_u

    def _erode_body_mask(self):
        """将body mask向内收缩，限制气体位移范围
        """
        # 3D形态学腐蚀操作
        struct = ndimage.generate_binary_structure(3, 1)
        struct = ndimage.iterate_structure(struct, self.body_erosion)
        eroded = ndimage.binary_erosion(self.body_mask, structure=struct)
        assert isinstance(eroded, np.ndarray)
        eroded = eroded.astype(np.bool_)
        return eroded

    def _identify_bowel_gas(self) -> np.ndarray:
        """识别肠道气体区域（解决阈值分割不完全问题）
        Returns:  gas_mask: 二值化气体掩码（1为气体区域，0为非气体）
        """
        # 1. 阈值分割
        initial_gas = (self.hu < self.hu_threshold).astype(np.uint8)
        initial_gas = initial_gas * self.body_mask
        # 3. 形态学闭运算补全小空洞（解决分割不完全），结构元素尺寸约3mm
        struct_size = int(np.ceil(3 / self.voxel_size))
        struct = ndimage.generate_binary_structure(3, 1)  # 3D结构元素
        struct = ndimage.iterate_structure(struct, struct_size)
        # 4. 闭运算（先膨胀后腐蚀）补全空洞，开运算去除小噪点
        gas_mask = ndimage.binary_closing(initial_gas, structure=struct)
        gas_mask = ndimage.binary_opening(gas_mask, structure=struct)
        # 5. 连通域分析，仅保留面积大于阈值的气体区域（排除微小噪点）
        # labeled, num_features = ndimage.label(gas_mask)
        # if num_features == 0:  # 无气体区域时直接返回空掩码
        #     return np.zeros_like(gas_mask, dtype=np.bool_)
        # sizes = ndimage.sum(gas_mask, labeled, range(1, num_features + 1))
        # min_size = 10  # 最小气体区域体素数（可调整）
        # mask_size = np.zeros(num_features + 1, dtype=bool)  # 长度为num_features+1
        # mask_size[1:] = sizes >= min_size  # 标签1~num_features的判断结果
        # # 现在labeled中的所有标签（0~num_features）都能正确索引mask_size
        # gas_mask = mask_size[labeled]
        return gas_mask.astype(np.bool_)

