# 自己实现拉普拉斯算子，对图像blurry_moon进行锐化。

import numpy as np
import cv2

#  ∇^2 f(x,y)=f(x+1,y)+f(x−1,y)+f(x,y+1)+f(x,y−1)−4f(x,y)
# 设定拉普拉斯算子 (3x3 卷积核)，k是用于调整的系数
kernel = np.array([[0, -1, 0],
                   [-1, 4 , -1],
                   [0, -1, 0]])

# 增益因子 (就是乘在计算结果上的倍数)来调整laplace处理效果的强弱
alpha = 0.7


def get_neighborhood(kernel, img, x, y):
    kernel_size = kernel.shape[0]
    offset = kernel_size // 2  # //整除
    return img[x - offset:x + kernel_size - offset, y - offset:y + kernel_size - offset]


def laplace(kernel, alpha, img):
    result = np.zeros_like(img)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            area = get_neighborhood(kernel, img, i, j)
            result[i, j] = img[i, j] + alpha * np.sum(area * kernel)

    # 裁剪值到[0, 255]并转换为uint8
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)


# 测试图片
def test_laplace_img(img_name, alpha_customize=1.0):
    print(f"begin the test for {img_name} with alpha = {alpha_customize}:")
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: {img_name} not found.")
        return
    laplace_img = laplace(kernel, alpha_customize, img)
    cv2.imshow(f"original image for {img_name}", img)
    # cv2.imshow(f"laplace to image for {img_name} with alpha = {alpha_customize}", laplace_img)
    cv2.imshow(f"alpha = {alpha_customize}", laplace_img)
    cv2.waitKey(0)
    print(f"  the test for {img_name} with alpha = {alpha_customize} end.")


# 测试
test_laplace_img("blurry_moon.tif", 1)
test_laplace_img("blurry_moon.tif", 0.7)
test_laplace_img("blurry_moon.tif", 0.3)
cv2.destroyAllWindows()
