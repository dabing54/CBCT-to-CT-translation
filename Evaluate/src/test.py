import os
import numpy as np
import matplotlib.pyplot as plt

data_path1 = r'E:\DaBing54\Desktop\data_ddim50\048.npy'
data_path2 = r'E:\DaBing54\Desktop\data_ddim50\050.npy'

img1 = np.load(data_path1)
img2 = np.load(data_path2)

diff = img1 - img2
plt.imshow(diff)
plt.show()

print(np.max(np.abs(diff)))
