import numpy as np  # 导入NumPy库，用于高效的数组和矩阵运算


def dual_gamma_transform(image, threshold, gamma1, gamma2):
    """
    对输入图像进行双γ分段指数变换，以增强对比度。

    根据Otsu方法计算的阈值，将图像的灰度值分为两个部分：
    1. 小于等于阈值的部分：使用γ1进行指数变换，增强背景区域。
    2. 大于阈值的部分：使用γ2进行指数变换，增强缺陷区域。

    参数：
        image (numpy.ndarray): 输入的灰度图像。
        threshold (float): 分段点的阈值，由Otsu方法计算得到。
        gamma1 (float): 用于背景部分的γ值，控制低灰度区域的对比度增强。
        gamma2 (float): 用于缺陷部分的γ值，控制高灰度区域的对比度增强。

    返回：
        numpy.ndarray: 经过双γ分段指数变换后的图像。
    """
    # 创建输入图像的副本，转换为浮点型以便进行指数运算
    result = np.copy(image).astype(np.float32)

    # 对灰度值小于等于阈值的部分，进行γ1指数变换
    # - 灰度值首先归一化到[0, 1]范围：result / 255
    # - 然后进行指数变换：result ** gamma1
    # - 再将值恢复到[0, 255]范围：* 255
    result[result <= threshold] = (result[result <= threshold] / 255) ** gamma1 * 255

    # 对灰度值大于阈值的部分，进行γ2指数变换
    # - 处理逻辑同上，但使用γ2增强高灰度区域的对比度
    result[result > threshold] = (result[result > threshold] / 255) ** gamma2 * 255

    # 返回结果图像，并将像素值类型转换回uint8（[0, 255]范围的8位整数）
    return np.uint8(result)
