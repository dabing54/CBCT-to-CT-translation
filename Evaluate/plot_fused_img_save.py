import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

idx = 40  # 40
patient_path = r'F:\2-MidData\sdfy_CBCT_to_CT\pelvic\252406'
save_root_path = r'E:\DaBing54\Desktop\out'
ct_path = os.path.join(patient_path, 'CT.nii.gz')
cbct_path = os.path.join(patient_path, 'CBCT.nii.gz')
dcbct_path = os.path.join(patient_path, 'dCBCT.nii.gz')
dpi = 600

ct = nib.load(ct_path).get_fdata().transpose(2, 1, 0)
cbct = nib.load(cbct_path).get_fdata().transpose(2, 1, 0)
dcbct = nib.load(dcbct_path).get_fdata().transpose(2, 1, 0)

ct = ct[idx]
cbct = cbct[idx]
dcbct = dcbct[idx]

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16



# 1
plt.figure(figsize=(5, 5))
plt.imshow(ct, cmap='gray', vmin=-160, vmax=240)
plt.axis('off')
save_path = os.path.join(save_root_path, 'ct.png')
plt.savefig(save_path, dpi=dpi)
plt.close()

# 2
plt.figure(figsize=(5, 5))
plt.imshow(cbct, cmap='gray', vmin=-160, vmax=240)
plt.axis('off')
save_path = os.path.join(save_root_path, 'cbct.png')
plt.savefig(save_path, dpi=dpi)
plt.close()

# 3
plt.figure(figsize=(5, 5))
plt.imshow(dcbct, cmap='gray', vmin=-160, vmax=240)
plt.axis('off')
save_path = os.path.join(save_root_path, 'deform_cbct.png')
plt.savefig(save_path, dpi=dpi)
plt.close()



# 图像归一，为了融合显示
def normalize_img(img):
    img = np.clip(img, -160, 240)
    img_min, img_max = np.min(img), np.max(img)
    img = (img - img_min) / (img_max - img_min)
    img = (img * 255).astype(np.uint8)
    return img

ct = normalize_img(ct)
cbct = normalize_img(cbct)
dcbct = normalize_img(dcbct)

h, w = ct.shape[0], ct.shape[1]
fused_cbct = np.zeros((h, w, 3), dtype=np.uint8)
fused_cbct[:,:,0] = ct  # 红
fused_cbct[:, :, 1] = cbct  # 绿

fused_dcbct = np.zeros((h, w, 3), dtype=np.uint8)
fused_dcbct[:,:,0] = ct
fused_dcbct[:, :, 1] = dcbct


# 4
plt.figure(figsize=(5, 5))
plt.imshow(fused_cbct)
plt.axis('off')
save_path = os.path.join(save_root_path, 'fused CBCT.png')
plt.savefig(save_path, dpi=dpi)
plt.close()

# 5
plt.figure(figsize=(5, 5))
plt.imshow(fused_dcbct)
plt.axis('off')
save_path = os.path.join(save_root_path, 'fused dCBCT.png')
plt.savefig(save_path, dpi=dpi)
plt.close()

