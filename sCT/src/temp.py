import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

num_bins = 60
series_name = 'Full Dose Images'
# series_name = 'sCBCT2'
data_root_path = r'F:\3-Data\1-SynthRADTask2'
patients = os.listdir(data_root_path)
patients = [item for item in patients if item.startswith('2PA')]
patients = patients[:10]

# mearged_hist = np.zeros(num_bins, dtype=np.int64)
# for patient in patients:
#     print(patient)
#     patient_path = os.path.join(data_root_path, patient)
#     series_path = os.path.join(patient_path, series_name)
#     items = os.listdir(series_path)
#     for item in tqdm(items):
#         img_path = os.path.join(series_path, item)
#         mask_path = img_path.replace(series_name, 'Mask')
#         img = np.load(img_path)
#         mask = np.load(mask_path)
#         arr = img[mask==1]
#         hist, bins = np.histogram(arr, bins=num_bins, range=(-1000, 1000))
#         mearged_hist += hist
#     # end for item
# # end for patient
# mearged_hist = mearged_hist / np.sum(mearged_hist)
# print(mearged_hist.round(2))
# plt.figure(figsize=(10, 6))
# plt.bar(bins[:-1], mearged_hist, width=20, align='edge')
# plt.xlim(-1024, 1024)
# plt.grid(axis='y', alpha=0.5)
# plt.show()

result = list()
for patient in patients:
    print(patient)
    patient_path = os.path.join(data_root_path, patient)
    series_path = os.path.join(patient_path, series_name)
    items = os.listdir(series_path)
    for item in tqdm(items):
        img_path = os.path.join(series_path, item)
        mask_path = img_path.replace(series_name, 'Mask')
        img = np.load(img_path)
        mask = np.load(mask_path)
        arr = img[mask==1]
        p1 = np.percentile(arr, 40)
        p2 = np.percentile(arr, 60)
        arr = arr[(arr<p2) & (arr>p1)]
        p = arr.mean()
        result.append(p)
    # end for
# end for
result = np.array(result)
print(result.mean())
plt.hist(result)
plt.show()