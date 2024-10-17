# 任务2
# 实现彩色图像的直方图均衡化（可以用OpenCV自带函数）
# 方法1：对RGB通道分别做直方图均衡化再合成
# 方法2：转换到HSV空间，仅对亮度分量V用直方图均衡化，再转换回RGB
# 用sky，mushroom两个图片来对比两个方法的结果

import cv2

# 方法1：对RGB通道分别做直方图均衡化再合成
def equalizeHist_rgb(image):
    # 拆分RGB三通道
    (b,g,r) = cv2.split(image)

    # 分别做直方图均衡化
    r = cv2.equalizeHist(r)
    g = cv2.equalizeHist(g)
    b = cv2.equalizeHist(b)

    # 合成结果图像
    result = cv2.merge((b,g,r))
    return result

# 方法2：转换到HSV空间，仅对亮度分量V用直方图均衡化，再转换回RGB
def equalizeHist_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 分离通道
    (h, s, v) = cv2.split(hsv)

    # 仅对亮度分量V用直方图均衡化
    v = cv2.equalizeHist(v)

    # 合成图像并转回成RGB
    result = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    return result

# 测试函数
def test_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)

    rgb = equalizeHist_rgb(image)
    hsv = equalizeHist_hsv(image)

    cv2.imshow("Equalized RGB", rgb)
    cv2.imshow("Equalized HSV", hsv)
    cv2.imshow("Original image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    test_image("sky.bmp")
    test_image("mushroom.png")
