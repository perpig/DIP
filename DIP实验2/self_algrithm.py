# coding=utf-8
import cv2
import numpy as np
import time

def resize(org, scale):  # scale：缩放因子
    org_h, org_w = org.shape[:2]  # 源图像宽高
    new_h = org_h * scale  # 目标图像高
    new_w = org_w * scale  # 目标图像宽

    # 遍历目标图像，插值
    new = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    for n in range(3):  # 3个通道的循环
        for dst_y in range(new_h):  
            for dst_x in range(new_w):  
                # 计算目标像素在原图片上的像素的坐标
                # Tips：原本应该是目标像素坐标 = 原像素坐标 * scale缩放因子
                # 但像素坐标是像素左上角，像素中心应该有0.5的偏移值 所以在乘缩放因子前后都要调整这个中心位置
                org_x = (dst_x + 0.5) / scale - 0.5
                org_y = (dst_y + 0.5) / scale - 0.5

                # 计算在源图上的左上角和右上角的x/y值
                # eg：(org_x_0, org_y_0)是左上角，(org_x_1, org_y_1)是右下角
                org_x_0 = int(np.floor(org_x))  # floor是下取整
                org_y_0 = int(np.floor(org_y))
                org_x_1 = min(org_x_0 + 1, org_w - 1)
                org_y_1 = min(org_y_0 + 1, org_h - 1)

                # 双线性插值
                # Tips：相减的距离作为衡量的权重
                # 对Y = org_y_0 这一行（上面的一行）做 X方向线性插值  选取的两个点：左上角 右上角
                value0 = (org_x_1 - org_x) * org[org_y_0, org_x_0, n] + (org_x - org_x_0) * org[org_y_0, org_x_1, n]
                # 对Y = org_y_1 这一行（上面的一行）做 X方向线性插值  选取的两个点：左下角 右下角
                value1 = (org_x_1 - org_x) * org[org_y_1, org_x_0, n] + (org_x - org_x_0) * org[org_y_1, org_x_1, n]
                # 对得到的value0 value1 做 Y方向的线性插值 这里的权重是上面和下面那一行的y值与此点的y值的距离来衡量
                new[dst_y, dst_x, n] = int((org_y_1 - org_y) * value0 + (org_y - org_y_0) * value1)

                # 此时这一个像素点已经计算得到它应该的值 继续循环 得到最后的图片

    return new


# test
if __name__ == '__main__':
    img_in = cv2.imread('test1.jpg')
    start = time.time()
    img_out = resize(img_in, 2)
    print('cost %f seconds' % (time.time() - start))

    cv2.imshow('src_image', img_in)
    cv2.imshow('dst_image', img_out)
    cv2.waitKey()
