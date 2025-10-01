import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import json
import pandas as pd

from scipy.ndimage import laplace
from skimage.metrics import structural_similarity
from skimage.measure import block_reduce
from skimage.transform import pyramid_gaussian
from skimage.feature import match_template
from sewar.full_ref import vifp


def main():
    data_root_path_base = r'F:\3-Data\1-SynthRADTask2'
    data_root_path_gen = r'F:\ProjectData\6-sCT3\SynthRADTask2\out'
    out_path = r'E:\DaBing54\Desktop\detal_result'
    series_name = 'FlowMatch_sCBCT1_200step'
    # series_name = 'Unet_sCBCT1'

    out_path = os.path.join(out_path, series_name + '.csv')
    # ----------------------------
    series_path = os.path.join(data_root_path_gen, series_name)
    patients = os.listdir(series_path)
    # patients = patients[:1]
    patient_num = len(patients)
    patient_out_list = list()
    for i, patient in enumerate(patients):
        print('Progress: %d/%d' % (i, patient_num))
        patient_path_gen = os.path.join(series_path, patient)
        patient_path_base = os.path.join(data_root_path_base, patient)
        patient_ct_path = os.path.join(patient_path_base, 'Full Dose Images')
        patient_mask_path = os.path.join(patient_path_base, 'Mask')
        items = os.listdir(patient_path_gen)
        out_list = list()
        for item in items:
            item_path = os.path.join(patient_path_gen, item)
            ct_path = os.path.join(patient_ct_path, item)
            mask_path = os.path.join(patient_mask_path, item)
            ct = np.load(ct_path)
            mask = np.load(mask_path)
            gen = np.load(item_path)
            # trans
            ct += 1024
            gen += 1024
            ct = np.clip(ct, 0, 2048)
            gen = np.clip(gen, 0, 2048)
            # out
            out = cal_metrics(ct, mask, gen)
            out_list.append(out)
        # end for item
        out = dict_mean(out_list)
        print(json.dumps(out, indent=4))
        patient_out_list.append(out)
    # end for patient
    df = pd.DataFrame(patient_out_list, index=patients)
    df.to_csv(out_path)
    out = dict_mean(patient_out_list)
    print('All:')
    print(json.dumps(out, indent=4))


def cal_metrics(ct, mask, gen, data_range=2048):
    assert ct.min() >= 0
    assert gen.min() >= 0
    mask = mask == 0  # mask转换
    gen_a = ma.masked_array(gen, mask=mask) if mask is not None else gen
    ct_a = ma.masked_array(ct, mask=mask) if mask is not None else ct
    gen[mask] = ct[mask]
    # mae， me
    diff = gen_a - ct_a
    mae = np.mean(np.abs(diff))
    me = np.mean(diff)
    rmse = np.sqrt(np.mean(diff**2))
    # psnr
    psnr = 20 * np.log10(data_range / rmse)
    # ssim
    # sewar包、pytorch-msssim包中计算的ssim不能去除mask外区域的值，且不能返回ssim map
    ssim = mask_ssim(gen, ct, mask, data_range)
    ms_ssim = mask_ms_ssim(gen, ct, mask, data_range, use_gauss=True)
    # vifp
    vifp_score = vifp(ct_a, gen_a)
    # ncc
    result = match_template(gen, ct, pad_input=True, mode='constant')
    ncc_score = result.max()
    # ## 单图像
    # Laplacian Variance
    laplacian_var = laplacian_variance(gen_a) / laplacian_variance(ct_a)
    # gradient magnitude mean
    grad_mag = gradient_magnitude_mean(gen_a) / gradient_magnitude_mean(ct_a)
    # High-Frequency Energy Ratio
    high_freq_energy_ratio = high_frequency_energy_ratio(gen_a) / high_frequency_energy_ratio(ct_a)
    # Grayscale Variance
    gray_var = np.var(gen_a) / np.var(ct_a)
    # Brenner梯度
    brenner = brenner_gradient(gen_a) / brenner_gradient(ct_a)
    # entropy
    entropy = img_entropy(gen_a) / img_entropy(ct_a)

    out = dict()
    out['mae'] = mae
    out['rmse'] = rmse
    out['me'] = me
    out['ssim'] = ssim
    out['ms_ssim'] = ms_ssim
    out['psnr'] = psnr
    out['vifp'] = vifp_score
    out['ncc'] = ncc_score
    out['laplacian_var'] = laplacian_var
    out['grad_mag'] = grad_mag
    out['high_freq_energy_ratio'] = high_freq_energy_ratio
    out['gray_var'] = gray_var
    out['brenner'] = brenner
    out['entropy'] = entropy
    return out

def mask_ssim(img1, img2, mask, data_range):
    _, ssim = structural_similarity(img1, img2, channel_axis=None, data_range=data_range, full=True)
    ssim = ma.masked_array(ssim, mask) if mask is not None else ssim
    ssim = ssim.mean()
    return ssim

def mask_ms_ssim(img1, img2, mask, data_range, use_gauss=True):
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]  # 典型权重（来自MS-SSIM论文）
    scales = len(weights)
    # 生成高斯金字塔
    if use_gauss:
        pyramid1 = list(pyramid_gaussian(img1, max_layer=scales - 1, downscale=2, preserve_range=True))
        pyramid2 = list(pyramid_gaussian(img2, max_layer=scales - 1, downscale=2, preserve_range=True))
        mask_pyramid = list(pyramid_gaussian(mask, max_layer=scales - 1, downscale=2, preserve_range=True))
    else:
        pyramid1 = pyramid_pool(img1, max_layer=scales-1, downscale=2)
        pyramid2 = pyramid_pool(img2, max_layer=scales - 1, downscale=2)
        mask_pyramid = pyramid_pool(mask, max_layer=scales - 1, downscale=2)
    # mask处理
    mask_pyramid = [(item > 0.5).astype(np.bool) for item in mask_pyramid]

    # 计算各尺度ssim并加权
    ms_ssim = 0.0
    for i, (im1, im2, mask_i) in enumerate(zip(pyramid1, pyramid2, mask_pyramid)):
        ssim = mask_ssim(im1, im2, mask_i, data_range)
        ms_ssim += weights[i] * ssim
    return ms_ssim


def pyramid_pool(img, max_layer, downscale=2):
    pyramid = [img]
    # 构建金字塔（仅平均池化）
    current_image = img.copy()
    for _ in range(max_layer - 1):
        # 使用 block_reduce 进行nxn平均池化
        pooled = block_reduce(current_image, block_size=(downscale, downscale), func=np.mean)
        pyramid.append(pooled)
        current_image = pooled
    return pyramid



def laplacian_variance(img):
    laplacian = laplace(img)
    laplacian_var = np.var(laplacian)
    return laplacian_var


def gradient_magnitude_mean(img):
    gx = np.gradient(img, axis=1)
    gy = np.gradient(img, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    grad_mag = np.mean(grad_mag)
    return grad_mag

def high_frequency_energy_ratio(img, radius=0.2):
    """计算高频能量比"""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    mask = (x - ccol)**2 + (y - crow)**2 > (radius * min(rows, cols)/2)**2
    magnitude_spectrum = np.abs(fshift) ** 2
    return np.sum(magnitude_spectrum * mask) / np.sum(magnitude_spectrum)


def brenner_gradient(img):
    # brenner梯度，对模糊敏感
    shifted = np.roll(img, -2, axis=0)
    brenner = np.mean((img - shifted) ** 2)
    return brenner


def img_entropy(img, bins=256):
    # 模糊图像熵较低
    hist, _ = np.histogram(img.flatten(), bins=bins, density=True)
    hist = hist[hist > 0]  # 去除零概率
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


def dict_mean(dict_list):
    assert isinstance(dict_list, list)
    keys = dict_list[0].keys()
    out = dict()
    for key in keys:
        val = [d[key] for d in dict_list]
        val = np.array(val).mean()
        val = np.round(val, 4)
        out[key] = val
    # end for key
    return out


if __name__ == '__main__':
    main()
