import matplotlib.pyplot as plt

def show_images(images, titles):
    """显示多张图像"""
    plt.figure(figsize=(12, 8))
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def save_comparison(image_list, title_list, output_path):
    """保存多张图像对比结果到文件"""
    plt.figure(figsize=(12, 8))
    for i, img in enumerate(image_list):
        plt.subplot(1, len(image_list), i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title_list[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def show_images2(images, titles):
    """显示每张图像，逐个显示"""
    for i, image in enumerate(images):
        plt.figure(figsize=(6, 6))  # 每次创建一个新的图像窗口
        plt.imshow(image, cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
        plt.show()  # 每张图像单独显示

