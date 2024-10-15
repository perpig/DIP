# 自己实现拉普拉斯算子，对图像blurry_moon进行锐化。

import numpy as np
import cv2

#  ∇^2 f(x,y)=f(x+1,y)+f(x−1,y)+f(x,y+1)+f(x,y−1)−4f(x,y)
# 设定拉普拉斯算子 (3x3 卷积核)
kernel = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])

def get_neighborhood(kernel, img, x, y):
    kernel_size = kernel.shape[0]
    offset = kernel_size // 2  # //整除
    return img[x - offset:x + kernel_size - offset, y - offset:y + kernel_size - offset]

def laplace(kernel,img):
    result = np.zeros_like(img)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            area = get_neighborhood(kernel, img, i, j)
            for c in range(img.shape[2]):
                result[i, j, c] =img[i, j, c] + np.sum(area[:, :, c] * kernel)

    # 裁剪值到[0, 255]并转换为uint8
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8)

# 测试图片
def test_img(img_name):
    print(f"begin the test for {img_name}:")
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: {img_name} not found.")
        return
    laplace_img = laplace(kernel, img)
    cv2.imshow(f"original image for {img_name}", img)
    cv2.imshow(f"laplace to image for {img_name}", laplace_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"  the test for {img_name} end.")


# 测试
test_img("blurry_moon.tif")
