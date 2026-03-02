import random

import numpy as np
import torch

import torchvision.tv_tensors as tv_tensors
from torchvision.transforms import v2
import torchvision.transforms.functional as F


class TrainTransform:
    def __init__(self, obj_size, trans_cfg):
        min_val = trans_cfg.min_val
        max_val = trans_cfg.max_val
        max_std = trans_cfg.max_std
        max_bright = trans_cfg.max_bright
        is_remove_out_mask = trans_cfg.is_remove_out_mask
        mid_size = np.around(np.array(obj_size) * 1.1, decimals=0).astype(int)
        mid_size2 = np.around(np.array(obj_size) * 1.2, decimals=0).astype(int)

        self.base_trans = v2.Compose([
            v2.ToImage(),  # mask 会忽略
            CenterPad(obj_size=mid_size, pad_val=min_val),
            # v2.RandomCrop(size=mid_size),
            # v2.RandomResize(min_size=obj_size, max_size=mid_size2),
            v2.CenterCrop(size=obj_size),
            v2.ToDtype(torch.float32)
        ])

        self.remove_out_mask = RemoveOutMask(min_val) if is_remove_out_mask else None
        self.random_noise = RandomNoise(max_std) if max_std > 0 else None
        self.random_bright = RandomBright(max_bright)
        self.clip_scale = ClipAndScale(min_val, max_val)

        self.is_use_hist_eq = trans_cfg.is_use_hist_eq

    def __call__(self, img1, img2, mask):
        mask = tv_tensors.Mask(mask)
        img1, img2, mask = self.base_trans(img1, img2, mask)

        if self.remove_out_mask is not None:
            img1 = self.remove_out_mask(img1, mask)
            img2 = self.remove_out_mask(img2, mask)
        if self.random_noise is not None:
            img2 = self.random_noise(img2)
            # img1 = self.random_noise(img1)
        if self.random_bright is not None:
            img2 = self.random_bright(img2)

        img1 = self.clip_scale(img1)
        img2 = self.clip_scale(img2)

        if self.is_use_hist_eq:
            img2 = histogram_equalization(img2)

        return img1, img2, mask


class TestTransform:
    def __init__(self, obj_size, trans_cfg):
        min_val = trans_cfg.min_val
        max_val = trans_cfg.max_val
        is_remove_out_mask = trans_cfg.is_remove_out_mask

        self.base_trans = v2.Compose([
            v2.ToImage(),  # mask 会忽略
            CenterPad(obj_size=obj_size, pad_val=min_val),
            v2.CenterCrop(size=obj_size),
            v2.ToDtype(torch.float32)
        ])

        self.remove_out_mask = RemoveOutMask(min_val) if is_remove_out_mask else None
        self.clip_scale = ClipAndScale(min_val, max_val)

        # self.mask_crop = CropImgByMask(obj_size)  # 若要使得mask位于图片中央，则用

        self.is_use_hist_eq = trans_cfg.is_use_hist_eq

    def __call__(self, img1, img2, mask):
        mask = tv_tensors.Mask(mask)
        img1, img2, mask = self.base_trans(img1, img2, mask)

        if self.remove_out_mask is not None:
            img1 = self.remove_out_mask(img1, mask)
            img2 = self.remove_out_mask(img2, mask)
        # end if
        img1 = self.clip_scale(img1)
        img2 = self.clip_scale(img2)

        if self.is_use_hist_eq:
            # img2 = histogram_equalization(img2)
            img2 = histogram_equalization_mask(img2, mask)

        return img1, img2, mask


class CenterPad:
    def __init__(self, obj_size, pad_val):
        self.obj_h, self.obj_w = obj_size, obj_size if isinstance(obj_size, int) else obj_size
        self.pad_val = pad_val

    def __call__(self, *args):
        size = args[0].shape[-2:]
        for img in args:  # 判断shape是否一致，忽略通道
            assert img.shape[-2:] == size
        # end for
        # 计算pad
        h, w = size[-2], size[-1]
        pad_l = pad_r = pad_u = pad_d = 0
        if self.obj_h > h:
            pad_u = (self.obj_h - h) // 2
            pad_d = self.obj_h - h - pad_u
        if self.obj_w > w:
            pad_l = (self.obj_w - w) // 2
            pad_r = self.obj_w - w - pad_l
        pad = [pad_l, pad_u, pad_r, pad_d]
        if not any(pad):
            return args
        # 执行
        out = list()
        for img in args:
            fill = 0 if isinstance(img, tv_tensors.Mask) else self.pad_val
            img = v2.functional.pad(img, padding=pad, fill=fill)
            out.append(img)
        # end for
        return out


class ClipAndScale:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        img = torch.clip(img, self.min_val, self.max_val)
        img = (img - self.min_val) / (self.max_val - self.min_val) * 2 - 1
        return img


class RandomNoise:
    def __init__(self, max_std):
        self.max_std = max_std

    def __call__(self, img):
        if self.max_std == 0:
            return img
        std = torch.rand(1) * self.max_std
        noise = torch.randn_like(img) * std
        img = img + noise
        return img


class RandomBright:
    def __init__(self, max_bright):
        self.max_bright_neg = max_bright[0]
        self.max_bright_pos = max_bright[1]

    def __call__(self, img):
        if self.max_bright_neg == 0 and self.max_bright_pos == 0:
            return img
        bright = random.randint(self.max_bright_neg, self.max_bright_pos)
        img = img + bright
        return img


class RemoveOutMask:
    def __init__(self, min_val):
        self.min_val = min_val

    def __call__(self, img, mask):
        img[:, mask == 0] = self.min_val  # img有c通道，mask无c通道
        return img


class CropImgByMask:
    """如果尺寸大于目标尺寸则裁剪，裁剪后mask位于中心"""
    def __init__(self, obj_size):
        self.obj_h, self.obj_w = obj_size, obj_size if isinstance(obj_size, int) else obj_size

    def __call__(self, img1, img2, mask):
        h, w = img1.shape
        hs, ws = 0, 0
        he, we = h, w
        if h > self.obj_h:
            line = np.sum(mask, axis=1)
            hs, he = self.cal_border_index(line, self.obj_h)
        if w > self.obj_w:
            line = np.sum(mask, axis=0)
            ws, we = self.cal_border_index(line, self.obj_w)
        img1 = img1[hs:he, ws:we]
        img2 = img2[hs:he, ws:we]
        mask = mask[hs:he, ws:we]
        return img1, img2, mask

    @ staticmethod
    def cal_border_index(line, obj_size):
        max_e = len(line)
        c = int(np.nonzero(line)[0].mean().round(0))
        s = c - obj_size // 2
        e = s + obj_size
        # 处理超出边界的问题
        if s < 0:
            e = e - s
            s = 0
        if e > max_e:
            s = s - e + max_e
            e = max_e
        return s, e

def histogram_equalization(img):
    assert img.ndim == 3 and img.shape[0] == 1  # c, h, w
    img_norm = ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8)

    img_eq = F.equalize(img_norm)
    img_eq = img_eq / 127.5 - 1  # [0, 255] -> [-1, 1]
    return img_eq

def histogram_equalization_mask(img, mask):
    assert img.ndim == 3 and img.shape[0] == 1  # c, h, w
    img = img[0]
    shape = img.shape
    img = img.numpy().astype(np.float32)
    img = (img + 1.0) / 2.0
    masked_img = img[mask > 0]
    hist, _ = np.histogram(masked_img, bins=256, range=(0, 1))
    cdf = hist.cumsum()
    cdf_norm = cdf / cdf[-1]
    img_eq = np.interp(img.flatten(), np.arange(0, 1, 1/256), cdf_norm)
    img_eq = img_eq.reshape(shape)
    img_eq = img_eq * 2 - 1
    img_eq = torch.from_numpy(img_eq).unsqueeze(0)
    img_eq = img_eq.to(torch.float32)
    return img_eq

