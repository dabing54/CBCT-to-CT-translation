import os
import numpy as np
import matplotlib.pyplot as plt

data_root_path = r'F:\3-Data\2-sdfy_cbct_to_ct\pelvic'
obj_slice = 10

def main():
    patients = os.listdir(data_root_path)
    for patient in patients:
        patient_path = os.path.join(data_root_path, patient)
        patient_ct_path = os.path.join(patient_path, 'CT')
        patient_dcbct_path = os.path.join(patient_path, 'dCBCT')
        patient_scbct_path = os.path.join(patient_path, 'sCBCT')
        items = os.listdir(patient_ct_path)
        obj_item_name = items[10]
        plot_3_img(patient_ct_path, patient_scbct_path, patient_dcbct_path, obj_item_name, patient)
        obj_item_name = items[-10]
        plot_3_img(patient_ct_path, patient_scbct_path, patient_dcbct_path, obj_item_name, patient)

def plot_3_img(patient_ct_path, patient_scbct_path, patient_dcbct_path, obj_item_name, patient):
    item_ct_path = os.path.join(patient_ct_path, obj_item_name)
    item_scbct_path = os.path.join(patient_scbct_path, obj_item_name)
    item_dcbct_path = os.path.join(patient_dcbct_path, obj_item_name)
    ct = np.load(item_ct_path)
    scbct = np.load(item_scbct_path)
    dcbct = np.load(item_dcbct_path)

    plt.figure(figsize=(15, 5))  # w, h
    manager = plt.get_current_fig_manager()
    # manager.window.wm_geometry(f"+100+10")  # w, h
    manager.window.move(10, 10)  # w, h
    plt.suptitle(patient + '--' + obj_item_name)

    plt.subplot(1, 3, 1)
    plt.imshow(ct, cmap='gray', vmin=-160, vmax=240)
    plt.title('CT')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(dcbct, cmap='gray', vmin=-160, vmax=240)
    plt.title('dCBCT')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(scbct, cmap='gray', vmin=-160, vmax=240)
    plt.title('sCBCT')
    plt.axis('off')

    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
