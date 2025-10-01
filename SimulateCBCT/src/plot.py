import matplotlib.pyplot as plt

def plot_two_3d_img(img3d1, img3d2, name1, name2, v_min=-1024, v_max=1024):
    num = img3d1.shape[0]
    assert img3d2.shape[0] == num
    for i in range(num):
        img1 = img3d1[i]
        img2 = img3d2[i]
        plt.figure(figsize=(12, 12)).canvas.manager.window.geometry("+300+100")
        plt.subplot(2, 2, 1)
        plt.title(name1 + '_%d/%d' % (i, num))
        plt.imshow(img1, cmap='gray', vmin=v_min, vmax=v_max)
        plt.subplot(2, 2, 2)
        plt.title(name2 + '_%d/%d' % (i, num))
        plt.imshow(img2, cmap='gray', vmin=v_min, vmax=v_max)
        plt.subplot(2, 2, 3)
        plt.title('diff_%d/%d' % (i, num))
        plt.imshow(img2 - img1, vmin=-50, vmax=50)
        plt.show()
        plt.close()
    # end for

