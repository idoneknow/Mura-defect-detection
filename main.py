import time
import cv2  # OpenCV库，用于图像处理
import numpy as np  # NumPy库，用于数组和矩阵运算
import yaml  # PyYAML库，用于加载配置文件
from utils.dct import apply_dct, apply_idct  # 导入DCT和IDCT相关函数
from utils.otsu import otsu_threshold  # 导入Otsu阈值分割函数
from utils.gamma_transform import dual_gamma_transform  # 导入双γ分段指数变换函数
from utils.evaluation import evaluate_detection_improved  # 导入评估检测结果的函数
from utils.visualization import show_images,show_images2  # 导入用于可视化的函数

t_start = time.time()
# 加载配置文件，读取用户设置的参数
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 加载输入图像，将其转换为灰度图像
input_image = cv2.imread("data/input_image1.png", cv2.IMREAD_GRAYSCALE)  ## cv2.IMREAD_GRAYSCALE: 这是一个标志，用于指定图像应该以灰度模式[0-255]读取
ground_truth = cv2.imread("data/ground_truth_mask1.png", cv2.IMREAD_GRAYSCALE)

# 步骤1：DCT处理和背景重建
# 对输入图像进行离散余弦变换（DCT），生成DCT系数矩阵
dct_coeffs = apply_dct(input_image)
# 对DCT系数进行逆变换（IDCT），生成重建的背景图像
background = apply_idct(dct_coeffs)

# 确保输入和背景图像的数据类型一致
input_image = input_image.astype(np.float32)  # 转换输入图像为浮点型
background = background.astype(np.float32)   # 转换背景图像为浮点型

# 步骤2：提取Mura缺陷区域
# 将原始图像减去背景图像，提取出Mura缺陷区域
mura_image = cv2.subtract(input_image, background)

# 将结果裁剪到[0, 255]范围，并转换回uint8格式，便于后续处理和保存
mura_image = np.clip(mura_image, 0, 255).astype(np.uint8)

# 步骤3：Otsu阈值分割和双γ分段指数变换
# 使用Otsu方法计算最佳阈值，用于分割Mura缺陷区域和背景
threshold = otsu_threshold(mura_image)
# 对Mura图像进行对比度增强，通过双γ分段指数变换分别处理背景和缺陷区域
enhanced_image = dual_gamma_transform(
    mura_image,
    threshold,
    config["parameters"]["gamma1"],  # 配置中指定的γ1值
    config["parameters"]["gamma2"]  # 配置中指定的γ2值
)
t_end = time.time()
use_time =round((t_end - t_start), 3)


# 步骤4：保存结果图像
# 保存重建的背景图像
cv2.imwrite("data/reconstructed_image.png", background)
# 保存对比度增强后的Mura检测图像
cv2.imwrite("data/output_mura_image.png", mura_image)
cv2.imwrite("data/output_mura_detected.png", enhanced_image)
# 设置标题和字体参数
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (255, 255, 255)  # 白色
line_type = 1
margin = 10  # 图片之间的间隔
title_height = 30  # 标题区域的高度

# 为每张图片添加标题
titles = ["Input Image","Background","Mura Image", "Enhanced Image"]
final_images = []

for i, image in enumerate([input_image, background,mura_image, enhanced_image]):
    # 创建标题区域的空白图像
    title_img = np.zeros((title_height, image.shape[1]), dtype=np.uint8)
    cv2.putText(title_img, titles[i], (10, int(0.5 * title_height)), font, font_scale, font_color, line_type)

    # 将标题区域和图片垂直拼接
    combined = np.vstack((title_img, image))
    final_images.append(combined)

# 水平拼接图片，每张图片之间添加一定距离
combined_image = final_images[0]
for img in final_images[1:]:
    combined_image = np.hstack((combined_image, np.zeros((img.shape[0], margin), dtype=np.uint8), img))

# 保存最终拼接的图片
cv2.imwrite("data/output_combined.png", combined_image)

# 步骤5：评估检测结果
'''加载人工标注的真值掩码图像,从json文件转为PNG文件（一次性）
# import json
# def get_image_size_from_annotations(json_path):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#
#     max_x, max_y = 0, 0
#     for annotation in data:
#         if "content" in annotation:
#             for point in annotation["content"]:
#                 max_x = max(max_x, point["x"])
#                 max_y = max(max_y, point["y"])
#
#     return int(max_y + 1), int(max_x + 1)
#
# def create_ground_truth_mask(json_path, image_size=None):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#
#     # 如果 image_size 为空，动态计算图像尺寸
#     if image_size is None:
#         image_size = get_image_size_from_annotations(json_path)
#
#     height, width = image_size
#     mask = np.zeros((height, width), dtype=np.uint8)
#
#     for annotation in data:
#         if "content" in annotation:
#             points = np.array([[point["x"], point["y"]] for point in annotation["content"]], dtype=np.int32)
#             # 裁剪坐标到图像尺寸
#             points = np.clip(points, [0, 0], [width - 1, height - 1])
#             cv2.fillPoly(mask, [points], color=255)
#
#     return mask
#
# # 示例：读取 JSON 并生成掩码
# json_path = "data/input_image4.json"
# image_size = (262,461)  # 假设图像尺寸为 512x512
# # image_size = None  # 如果需要动态计算尺寸，则传入 None；否则传入固定尺寸 (height, width)
# ground_truth = create_ground_truth_mask(json_path, image_size)
# # 保存掩码为图像文件（可选）
# cv2.imwrite("data/ground_truth_mask4.png", ground_truth)
'''

metrics = evaluate_detection_improved(input_image, enhanced_image,use_time,ground_truth)
print("Improved Evaluation Metrics:", metrics)



# 步骤6：可视化结果
# 显示原始图像、背景图像、Mura检测图像和对比度增强后的图像
show_images(
    [input_image, background, mura_image, enhanced_image],
    ["Original Mura", "Background", "Mura", "Enhanced"]
)
# show_images2([input_image, background, mura_image, enhanced_image],
#             ["Original Mura", "Background", "Mura", "Enhanced"])


# 输出评估指标到控制台
print("Evaluation Metrics:", metrics)
