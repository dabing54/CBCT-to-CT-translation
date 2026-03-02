import SimpleITK as sitk

def rigid_registration(fixed_img, moving_img, is_3d, return_transform=False):
    assert isinstance(fixed_img, sitk.Image)
    assert isinstance(moving_img, sitk.Image)
    dim = "3" if is_3d else "2"
    elastix_filter = sitk.ElastixImageFilter()
    # elastix_filter.LogToConsoleOn()  # 开启日志
    elastix_filter.LogToConsoleOff()  # 不显示过程日志
    elastix_filter.SetFixedImage(fixed_img)
    elastix_filter.SetMovingImage(moving_img)

    param_map = sitk.ParameterMap()
    # ========== **********ImageTypes********** ==========
    param_map["FixedInternalImagePixelType"] = ["float"]
    param_map["MovingInternalImagePixelType"] = ["float"]
    param_map["FixedImageDimension"] = [dim]
    param_map["MovingImageDimension"] = [dim]
    param_map["UseDirectionCosines"] = ["true"]
    param_map["NumberOfThreads"] = ["16"]
    # ========== *********Components********** ==========
    param_map["Registration"] = ["MultiResolutionRegistration"]
    param_map["Interpolator"] = ["LinearInterpolator"]
    param_map["Resampler"] = ["DefaultResampler"]
    param_map["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
    param_map["FixedImagePyramid"] = ["FixedRecursiveImagePyramid"]
    param_map["MovingImagePyramid"] = ["MovingRecursiveImagePyramid"]
    # ==========刚性变换（EulerTransform）、采样器、优化器、度量
    param_map["Transform"] = ["EulerTransform"]
    param_map["Sampler"] = ["RandomCoordinate"]
    param_map["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    param_map["Metric"] = ["AdvancedMattesMutualInformation"]

    # ========== ************Transform************ ==========
    param_map["AutomaticScalesEstimation"] = ["true"]
    param_map["AutomaticTransformInitialization"] = ["true"]
    param_map["HowToCombineTransforms"] = ["Compose"]
    param_map["AutomaticTransformInitializationMethod"] = ["CenterOfMass"]

    # ========== ************Similarity Measure************ ==========
    param_map["NumberOfHistogramBins"] = ["16", "32", "32", "64"]

    # ========== **************Multiresolution ********************** ==========
    param_map["NumberOfResolutions"] = ["4"]
    param_map["ImagePyramidSchedule"] = ["8 8 8", "4 4 4", "2 2 2", "1 1 1"]
    param_map["MaximumNumberOfIterations"] = ["500", "300", "200", "200"]

    # ========== ************Image Sampling************ ==========
    param_map["NumberOfSpatialSamples"] = ["2048"]
    param_map["ImageSampler"] = ["Grid"]
    param_map["SampleGridSpacing"] = ["4", "4", "4", "2"]
    param_map["MaximumNumberOfSamplingAttempts"] = ["20"]
    param_map["RequiredRatioOfValidSamples"] = ["0.045"]
    param_map["FixedImageBSplineInterpolationOrder"] = ["1"]

    # ========== ************* Interpolation and Resampling **************** ==========
    param_map["BSplineInterpolationOrder"] = ["1"]
    param_map["FinalBSplineInterpolationOrder"] = ["3"]
    param_map["DefaultPixelValue"] = ["-1024"]

    # ========== ************Output*************** ==========
    param_map["WriteResultImage"] = ["false"]
    param_map["WriteIterationInfo"] = ["false"]
    param_map["WriteTransformParameters"] = ["false"]
    param_map["WriteOriginalTransform"] = ["false"]
    param_map["WriteInterpolator"] = ["false"]

    elastix_filter.SetParameterMap(param_map)
    elastix_filter.Execute()
    img = elastix_filter.GetResultImage()  # sitk对象
    transform_params = elastix_filter.GetTransformParameterMap()[0]
    if return_transform:
        return img, transform_params
    else:
        return img
    # end

def deform_registration(fixed_img, moving_img, is_3d):
    dim = "3" if is_3d else "2"
    # 1. Elastix配准器
    elastix_filter = sitk.ElastixImageFilter()
    elastix_filter.LogToConsoleOff()  # 不显示过程日志
    # elastix_filter.LogToConsoleOn()
    elastix_filter.SetFixedImage(fixed_img)
    elastix_filter.SetMovingImage(moving_img)
    # 2. 参数配置
    param_map = sitk.ParameterMap()  # 空参数表，避免默认参数干扰
    # 内部计算像素类型
    param_map["FixedInternalImagePixelType"] = ["float"]
    param_map["MovingInternalImagePixelType"] = ["float"]
    param_map["FixedImageDimension"] = [dim]
    param_map["MovingImageDimension"] = [dim]
    param_map["UseDirectionCosines"] = ["true"]
    param_map["NumberOfThreads"] = ["16"]  # 根据CPU核心数调整，加速计算
    # ================================= *********Components********** =================================
    param_map["Registration"] = ["MultiResolutionRegistration"]  # MultiMetricMultiResolutionRegistration
    param_map["Interpolator"] = ["LinearInterpolator"]
    param_map["Resampler"] = ["DefaultResampler"]
    param_map["ResampleInterpolator"] = ["FinalBSplineInterpolator"]
    param_map["FixedImagePyramid"] = ["FixedRecursiveImagePyramid"]
    param_map["MovingImagePyramid"] = ["MovingRecursiveImagePyramid"]
    param_map["Transform"] = ["BSplineTransform"]  # B样条形变核心
    param_map["Sampler"] = ["RandomCoordinate"]
    param_map["Optimizer"] = ["AdaptiveStochasticGradientDescent"]  # ASGD优化器
    param_map["Metric"] = ["AdvancedMattesMutualInformation"]
    # param_map["Metric"] = ["AdvancedMattesMutualInformation", "AdvancedNormalizedCorrelation"]  # 互信息度量    AdvancedMattesMutualInformation  AdvancedNormalizedCorrelation
    # param_map['MetricWeight'] = ['0.7', '0.3']
    # ================================= ************Transform************ =================================
    param_map["AutomaticScalesEstimation"] = ["true"]
    param_map["AutomaticTransformInitialization"] = ["true"]
    param_map["HowToCombineTransforms"] = ["Compose"]
    param_map["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]  # 重心法初始化变换
    param_map["FinalGridSpacingInPhysicalUnits"] = ["10.0", "10.0", "10.0"]  # 初始粗网格间距（单位：mm），允许大范围形变
    # param_map["GridSpacingSchedule"] = ["12.0 12.0 12.0", "10.0 10.0 10.0", "8.0 8.0 8.0", "6.0 6.0 6.0"]  # 随分辨率细化逐渐加密网格
    # param_map["FinalGridSpacingInPhysicalUnits"] = ["10.0", "10.0", "10.0"]  # 初始粗网格间距（单位：mm），允许大范围形变
    # param_map["GridSpacingSchedule"] = ["10.0 10.0 10.0", "8.0 8.0 8.0", "6.0 6.0 6.0", "4.0 4.0 4.0"]
    # param_map["SplineOrder"] = ["3"]

    # ================================= ************Similarity Measure************ =================================
    param_map["NumberOfHistogramBins"] = ["16", "32", "32", "64"]  # （4层分辨率对应4个值）

    # ================================= **************Multiresolution ********************** =================================
    param_map["NumberOfResolutions"] = ["4"]  # 4层分辨率（原参数核心设置）
    param_map["ImagePyramidSchedule"] = ["8 8 8", "4 4 4", "2 2 2", "1 1 1"]  # （4层，每层3个值：x/y/z）
    # 原参数中该部分为注释，暂不配置
    # param_map["FinalGridSpacingInVoxels"] = ["1.0", "1.0", "1.0"]
    # param_map["FinalGridSpacingInPhysicalUnits"] = ["1.0", "1.0", "1.0"]
    # param_map["GridSpacingSchedule"] = ["6.0 6.0 6.0", "3.0 3.0 3.0", "1.0 1.0 1.0"]

    # ================================= ******************* Optimizer **************************** =================================
    param_map["MaximumNumberOfIterations"] = ["1200", "1000", "500", "300"]  # （4层分辨率对应迭代次数）
    # param_map["MaximumStepLength"] = ["4.0", "2.0", "1.0", "0.5"]  # 每层最大步长递减
    # param_map["MinimumStepLength"] = ["0.01"]  # 防止步长过小导致早停

    # ================================= ************Image Sampling************ =================================
    param_map["NumberOfSpatialSamples"] = ["2048"]  # 空间采样数  # 4096
    param_map["ImageSampler"] = ["Grid"]  # 网格采样  # Grid TODO 网格膀胱更好
    param_map["SampleGridSpacing"] = ["4", "4", "4", "2"]  #
    param_map["MaximumNumberOfSamplingAttempts"] = ["20"]  # 最大采样尝试次数20
    param_map["RequiredRatioOfValidSamples"] = ["0.045"]  # 有效采样比例
    param_map["FixedImageBSplineInterpolationOrder"] = ["1"]  # 固定图像B样条插值阶数

    # ================================= ************* Interpolation and Resampling **************** =================================
    param_map["BSplineInterpolationOrder"] = ["1"]  # 基础B样条插值阶数
    param_map["FinalBSplineInterpolationOrder"] = ["3"]  # 最终重采样插值阶数（原参数重点：提高精度）
    param_map["DefaultPixelValue"] = ["-1024"]  # 外插区域像素值（CT空气值，原参数核心）
    param_map["SmoothPositionInPhysicalUnits"] = ["true"]  # 平滑边界形变
    # =====信息======
    param_map["WriteIterationInfo"] = ["false"]
    param_map["WriteTransformParameters"] = ["true"]  # 保留变换参数文件（按需调整）
    param_map["WriteOriginalTransform"] = ["false"]  # 关闭原始变换文件
    param_map["WriteInterpolator"] = ["false"]  # 关闭插值器信息文件

    # 3.执行
    elastix_filter.SetParameterMap(param_map)
    elastix_filter.Execute()
    result = elastix_filter.GetResultImage()
    return result

def apply_transform(img, transform_params, is_mask=True):
    if is_mask:
        transform_params["Interpolator"] =["NearestNeighborInterpolator"]  # Mask必须用NearestNeighbor插值
        transform_params["FinalInterpolator"] = ["NearestNeighborInterpolator"]
        transform_params["DefaultPixelValue"] = ["0"]
        transform_params["ResultImagePixelType"] = ["unsigned_char"]  # unsigned_char
    transformix_filter = sitk.TransformixImageFilter()
    transformix_filter.LogToConsoleOff()  # 不显示过程日志
    transformix_filter.SetMovingImage(img)
    transformix_filter.SetTransformParameterMap(transform_params)
    transformix_filter.Execute()
    img = transformix_filter.GetResultImage()
    if is_mask:  # TODO 应用最邻近插值后仍然有1.00几或0.99几等小数，尚未解决
        img = sitk.BinaryThreshold(img, lowerThreshold=0.5,upperThreshold=float('inf'), insideValue=1, outsideValue=0)
        img = sitk.Cast(img, sitk.sitkUInt8)
    return img

def print_param(param):
    for key, value in param.items():
        value_str = ', '.join(value)
        print(f'{key}: {value_str}')
    # end for
