import cv2
import numpy as np


def evaluate_detection_improved(original, detected,use_time,ground_truth):
    """
    改进的缺陷检测评价函数
    Args:
        original (numpy.ndarray): 原始图像
        detected (numpy.ndarray): 检测图像
    Returns:
        dict: 各评价指标结果
    """
    results = {}
    results["use_time"] = use_time

    # Otsu 阈值分割
    _, binary_mask = cv2.threshold(detected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_mask = binary_mask > 0  # 转换为布尔掩码

    # 计算前景（缺陷）和背景的平均像素值
    foreground_pixels = detected[binary_mask]
    background_pixels = detected[~binary_mask]

    if len(foreground_pixels) > 0 and len(background_pixels) > 0:
        # 缺陷区域对比度提升
        contrast_foreground = np.mean(foreground_pixels)
        contrast_background = np.mean(background_pixels)
        results["defect_contrast"] = round(contrast_foreground - contrast_background,4)

        # 缺陷区域面积比例
        results["defect_area_ratio"] = round(len(foreground_pixels) / detected.size , 6)

    else:
        results["defect_contrast"] = "Insufficient Data"
        results["defect_area_ratio"] = "Insufficient Data"

    # 增强图像的动态范围
    results["dynamic_range"] = round(detected.max() - detected.min(),4)

    # **新增：识别正确率计算**
    # 将检测结果和人工标注的真值掩码都转换为布尔数组
    detected_binary = binary_mask  # 检测结果布尔掩码
    ground_truth_binary = ground_truth > 0  # 真值掩码布尔化（255为缺陷区域）

    # 计算真阳性、假阳性、真阴性、假阴性
    true_positive = np.sum(detected_binary & ground_truth_binary)  # 检测到缺陷且确实是缺陷
    true_negative = np.sum(~detected_binary & ~ground_truth_binary)  # 检测到背景且确实是背景
    false_positive = np.sum(detected_binary & ~ground_truth_binary)  # 检测到缺陷但实际上是背景
    false_negative = np.sum(~detected_binary & ground_truth_binary)  # 检测到背景但实际上是缺陷

    # 计算正确率
    total_pixels = detected.size  # 图像的总像素数
    accuracy = round((true_positive + true_negative) / total_pixels, 6)
    results["accuracy"] = accuracy

    return results

