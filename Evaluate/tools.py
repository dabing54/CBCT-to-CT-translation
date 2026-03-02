import numpy as np

def center_crop(img, obj_img):
    shape = np.array(img.shape)
    obj_shape = np.array(obj_img.shape)

    diff = shape - obj_shape
    half = diff // 2

    hs, ws = half[0], half[1]
    he= hs + obj_shape[0]
    we = ws + obj_shape[1]

    img = img[hs:he, ws:we]
    return img


