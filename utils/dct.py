import numpy as np  # 导入NumPy库，用于数组操作
import cv2  # 导入OpenCV库，用于图像处理


def apply_dct(image):
    """
    对输入的图像进行离散余弦变换（DCT），并保留低频直流系数。

    参数：
        image (numpy.ndarray): 输入的灰度图像。

    返回：
        numpy.ndarray: DCT系数矩阵，其中仅保留低频分量。
    """
    # 对图像进行DCT变换，转换到频域
    # zy:DCT能够将图像数据转换到一个能量更加集中的域，从而便于压缩,通常在浮点数上进行计算，因为这样可以提供更好的数值稳定性和精度
    # cv2.dct():用于计算一维或二维的离散余弦变换
    # DCT变换后的系数可能会有正有负，而且由于变换的特性，大多数系数会集中在数组的左上角（对于二维DCT），这表示图像的能量主要集中在低频部分
    # 低频成分代表了图像中缓慢变化的区域
    dct_coeffs = cv2.dct(np.float32(image))

    # 保留频域中低频部分的直流分量
    # 将第一行以外的所有行置零，仅保留第一行
    dct_coeffs[1:, :] = 0
    # 将第一列以外的所有列置零，仅保留第一列
    dct_coeffs[:, 1:] = 0

    # 返回处理后的DCT系数矩阵
    return dct_coeffs


def apply_idct(dct_coeffs):
    """
    基于DCT系数进行背景重建，通过逆DCT（IDCT）从频域回到空间域。

    参数：
        dct_coeffs (numpy.ndarray): DCT系数矩阵，其中仅保留低频分量。

    返回：
        numpy.ndarray: 重建的背景图像。
    """
    # 对DCT系数矩阵进行逆DCT变换，重建空间域中的背景图像
    # zy:在这种情况下，dct_coeffs 实际上只包含了图像的直流（DC）分量，而没有交流（AC）分量
    # 直流（DC）分量：这是图像的DCT变换中的第一个系数，代表了图像的平均亮度或灰度水平。它是一个图像的常数分量，即整个图像的亮度水平。
    # 交流（AC）分量：这些是DCT变换中的其他系数，代表了图像中的频率变化，如边缘、纹理等细节信息。如果这些分量为0，意味着图像中没有这些频率的变化
    # 当使用 cv2.idct 对这样的 dct_coeffs 矩阵进行逆变换时，你实际上是在重建图像的灰度图
    # 逆变换后得到的图像将是一个均匀的灰度图，其灰度值由DC分量决定
    return cv2.idct(dct_coeffs)
