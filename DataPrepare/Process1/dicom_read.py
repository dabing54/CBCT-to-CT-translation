import numpy as np
import pydicom
import cv2
import SimpleITK as sitk

def get_dcm_img(ct_path):
    """从CT文件夹中读取ct序列，仅支持文件夹中为单个序列"""
    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(ct_path)
    assert len(series_ids) == 1
    series_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(ct_path, series_ids[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_files)
    img = series_reader.Execute()
    return img  # sitk类型

def get_dcm_mask(rs_path, roi_names, shape, origin, space):
    rs = pydicom.dcmread(rs_path)
    # 获取目标ROI名字所对应的ROI Number
    roi_sequence = rs.StructureSetROISequence
    roi_numbers = [it.ROINumber for it in roi_sequence if it.ROIName in roi_names]
    if len(roi_numbers) == 0:
        print([it.ROIName for it in roi_sequence])
        raise Exception(rs_path + '文件中不存在目标名为' + roi_names + '的ROI')
    # 获取存储目标ROI点序列的结构
    roi_sequence = rs.ROIContourSequence
    contour_sequences = [it.ContourSequence for it in roi_sequence if it.ReferencedROINumber in roi_numbers]
    contour_sequence = list()
    for it in contour_sequences:  # 此步骤是为了兼容需同时考虑两个ROI的情况
        contour_sequence += it  # +是两个列表连接
    # 遍历每一个层面（也许一个层面有两个闭合区域）
    mask = np.zeros((shape[2], shape[1], shape[0]), dtype=np.uint8)  # d, h, w
    for contour in contour_sequence:
        contour_data = contour.ContourData
        # 将轮廓点转换为mask
        mask_2d, z = contour_point_to_mask(contour_data, shape, origin, space)
        z_index = cal_z_index(origin[2], space[2], z)
        mask[z_index] = np.bitwise_or(mask[z_index], mask_2d)
    # end for
    return mask  # np类型， d, h, w

def get_dcm_img_mask(ct_path, rs_path, roi_names, max_connected_domain='none'):
    """同时读取CT和对应的mask"""
    img = get_dcm_img(ct_path)  # sitk类型
    shape = img.GetSize()  # w, h, d
    origin = img.GetOrigin()
    space = img.GetSpacing()
    direction = img.GetDirection()
    assert direction == (1, 0, 0, 0, 1, 0, 0, 0, 1)
    # if direction[-1] == -1:  # TODO 不知道为什么PACS导出的CT，通过2.4.0版本的SimpleITK读取后direction为-1，通过2.2.1版本的SimpleITK读取后direction为1
    #     direction = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    #     img.SetDirection(direction)
    idx = int((shape[-1] - 1) * direction[-1])
    point = img.TransformIndexToPhysicalPoint((0, 0, idx))
    min_z, max_z = origin[-1], point[-1]
    assert int(round((max_z - min_z) / space[-1])) + 1 == shape[-1]
    mask_np = get_dcm_mask(rs_path, roi_names, shape, origin, space)
    if max_connected_domain == '3d':
        mask_np = get_max_connected_domain_3d(mask_np)
    elif max_connected_domain == '2d':
        mask_np = get_max_connected_domain_2d(mask_np)
    else:
        raise NotImplementedError(max_connected_domain)
    mask = sitk.GetImageFromArray(mask_np)
    mask.SetOrigin(origin)
    mask.SetSpacing(space)
    mask.SetDirection(direction)
    img = sitk.Cast(img, sitk.sitkInt16)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    return img, mask  # sitk对象


def contour_point_to_mask(contour_data, shape, origin, space):
    """将RS文件中的一个ROI的一个层面(一个闭合轮廓)数据转换为mask"""
    contour_data = np.array(contour_data)
    x = np.around((contour_data[::3] - origin[0]) / space[0])
    y = np.around((contour_data[1::3] - origin[1]) / space[1])
    z = round(contour_data[2], 1)  # z 保留1位小数, 因为单位为mm
    pts = np.vstack([x, y])
    pts = np.transpose(pts, (1, 0)).astype(int)  # [[x1,y1],[x2,y2]...]
    mask = np.zeros((shape[1], shape[0]), np.uint8)
    cv2.fillPoly(mask, [pts], color=1)  # color=(255, 255, 255), 经测试，color可用单通道值
    return mask, z  # mask为np

def cal_z_index(origin_z, space_z, z):
    z_index = (z - origin_z) / space_z
    z_index = int(round(z_index))
    return z_index

def get_max_connected_domain_3d(img_np):
    """对3d图像进行保留最大连通域操作"""
    img = sitk.GetImageFromArray(img_np)
    cc_filter = sitk.ConnectedComponentImageFilter()
    # cc_filter.SetFullyConnected(True)  # 26连通
    labbed_img = cc_filter.Execute(img)
    labbed_np = sitk.GetArrayFromImage(labbed_img)
    areas = np.bincount(labbed_np.flatten())
    areas = areas[1:]  # 标记0为背景
    max_label = np.argmax(areas) + 1
    mask_np = (labbed_np == max_label).astype(np.uint8)
    out = mask_np * img_np
    return out


def get_max_connected_domain_2d(img_np_3d):
    """对3d图像的每一个层面保留最大连通域， 输入为d, h, w"""
    cc_filter = sitk.ConnectedComponentImageFilter()
    mask_3d = np.zeros_like(img_np_3d)
    for i in range(img_np_3d.shape[0]):
        img = sitk.GetImageFromArray(img_np_3d[i])
        labbed_img = cc_filter.Execute(img)
        labbed_np = sitk.GetArrayFromImage(labbed_img)
        areas = np.bincount(labbed_np.flatten())
        areas = areas[1:]  # 标记0为背景
        if len(areas) > 0:
            max_label = np.argmax(areas) + 1
            mask_3d[i] = (labbed_np == max_label).astype(np.uint8)
    out = mask_3d * img_np_3d
    return out
