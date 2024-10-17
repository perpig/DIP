# 将araras.jpg的三通道拆开用灰度图像显示
import cv2

def channel_to_gray(filename):
    image = cv2.imread(filename, cv2.IMREAD_COLOR)

    # RGB 通道的名字
    channels = ['Red', 'Green', 'Blue']

    # 遍历 RGB 通道
    for i, channel in enumerate(channels):
        temp = image[:, :, i]  # 获取每个通道的图像
        gray_image = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)  # 将单通道转为3通道灰度图
        cv2.imshow(channel, gray_image)  # 使用通道名字作为窗口名

    cv2.imshow('Original Image', image)  # 显示原图
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    channel_to_gray("araras.jpg")
