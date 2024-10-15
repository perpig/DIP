# 自己实现均值滤波，中值滤波，并在space，mona，pcb上进行测试。
import cv2
import numpy as np

# 设定卷积大小
kernel_size = 3


# 获取kernal大小的点(x,y)周围的区域
def get_neighborhood(kernel_size, img, x, y):
    offset = kernel_size // 2  # //整除
    return img[x - offset:x + kernel_size - offset, y - offset:y + kernel_size - offset]


# 均值滤波实现
def average_filtering(img):
    result = np.zeros_like(img)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            # 将这个像素点的值替换成获取这个点周围区域计算的平均值
            result[i, j] = np.mean(get_neighborhood(kernel_size, img[:, :], i, j))
    return result


# 中值滤波实现
def median_filtering(img):
    result = np.zeros_like(img)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            result[i, j] = np.median(get_neighborhood(kernel_size, img[:, :], i, j))
    return result


# 测试图片
def test_img(img_name):
    print(f"begin the test for {img_name}:")
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: {img_name} not found.")
        return
    avg = average_filtering(img)
    med = median_filtering(img)
    cv2.imshow(f"original image for {img_name}", img)
    cv2.imshow(f"average_filtering for {img_name}", avg)
    cv2.imshow(f"median_filtering for {img_name}", med)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"  the test for {img_name} end.")


# 测试
test_img("Space.jpg")
test_img("Mona.jpg")
test_img("pcb.png")
