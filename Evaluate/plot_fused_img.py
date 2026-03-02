import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

idx = 40  # 40
patient_path = r'F:\2-MidData\sdfy_CBCT_to_CT\pelvic\252406'
ct_path = os.path.join(patient_path, 'CT.nii.gz')
cbct_path = os.path.join(patient_path, 'CBCT.nii.gz')
dcbct_path = os.path.join(patient_path, 'dCBCT.nii.gz')

ct = nib.load(ct_path).get_fdata().transpose(2, 1, 0)
cbct = nib.load(cbct_path).get_fdata().transpose(2, 1, 0)
dcbct = nib.load(dcbct_path).get_fdata().transpose(2, 1, 0)

ct = ct[idx]
cbct = cbct[idx]
dcbct = dcbct[idx]

plt.figure(figsize=(15, 10))  # w, h
manager = plt.get_current_fig_manager()
manager.window.wm_geometry(f"+100+10")  # w, h
# manager.window.move(10, 10)  # w, h

plt.subplot(2, 3, 1)
plt.imshow(ct, cmap='gray', vmin=-160, vmax=240)
plt.title('CT')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(cbct, cmap='gray', vmin=-160, vmax=240)
plt.title('CBCT')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(dcbct, cmap='gray', vmin=-160, vmax=240)
plt.title('deform CBCT')
plt.axis('off')

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


plt.subplot(2, 3, 5)
plt.imshow(fused_cbct)
plt.title('fused: CBCT(red) - CT(green)')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(fused_dcbct)
plt.title('fused: deform CBCT - CT ')
plt.axis('off')
plt.show()
plt.close()


