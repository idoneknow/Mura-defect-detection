import cv2  # 导入OpenCV库，用于图像处理


def otsu_threshold(image):
    """
    使用Otsu方法对输入图像计算全局阈值。

    Otsu方法是一种自适应的阈值分割技术，
    通过最大化类间方差自动确定图像的最佳分割阈值。

    参数：
        image (numpy.ndarray): 输入的灰度图像。

    返回：
        float: 由Otsu方法计算的全局阈值。
    """
    # 使用OpenCV的threshold函数结合Otsu方法计算阈值
    # 参数解释：
    # - `0`: 手动阈值，此处不使用，Otsu方法会自动计算
    # - `255`: 阈值后的最大值（对应二值化后像素值为255）
    # - `cv2.THRESH_BINARY + cv2.THRESH_OTSU`: 使用Otsu自动阈值方法

    # zy:cv2.threshold函数来对一个图像进行二值化处理，并且自动计算二值化的阈值，使用的方法是基于大津法（Otsu's method）
    # cv2.THRESH_BINARY + cv2.THRESH_OTSU: 这是两个标志的组合，用于指定二值化的类型和阈值计算方法
    # THRESH_BINARY: 指定二值化类型为二进制二值化，即高于阈值的像素设置为最大值，低于阈值的像素设置为0
    # THRESH_OTSU: 指定使用Otsu's method自动计算阈值。Otsu's method通过最小化类内方差或等价地最大化类间方差来确定最优阈值
    # _, threshold: 这是函数的返回值,cv2.threshold函数返回两个值：二值化后的图像和计算出的阈值。
    # 在这里，我们只关心阈值，所以使用_（一个惯用的方式，表示我们不关心这个返回值）来接收二值化后的图像，而threshold变量将存储计算出的阈值
    _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 返回计算得到的全局阈值
    return threshold
