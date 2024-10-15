# 任务1
# 自己实现一个Gamma变换，采用2种不同实现，使用查找表和不使用查找表，比较二者的效率差异。至少在
# light，dark两个图片上进行测试。

import cv2
import numpy as np
import time

# gamma公式： Output = 225 * exp((Input / 225) , 1 / gamma)

def gamma_LUT(image, gamma):
    # 使用查找表的Gamma变换
    # 查找表使用方法： result_image = cv2.LUT(image, table)
    # 使用逆gamma值方便计算
    inv_gamma = 1.0 / gamma
    # 创建表
    table = np.array([225 * (i / 255.0) ** inv_gamma for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def gamma_no_LUT(image, gamma):
    # 不使用查找表的Gamma变换
    # 使用逆gamma值方便计算
    inv_gamma = 1.0 / gamma
    return np.array(255 * (image / 255.0) ** inv_gamma, dtype='uint8')

# main( )

light = cv2.imread("light.tif")
dark = cv2.imread("dark.jpg")

# 设定gamma值
gamma_light = 0.4 # 变暗
gamma_dark = 3 # 变亮

# light图片
# 有LUT
start_time = time.time()
result_LUT_light = gamma_LUT(light,gamma_light)
time_LUT_light = time.time() - start_time
# 无LUT
start_time = time.time()
result_no_LUT_light = gamma_no_LUT(light,gamma_light)
time_no_LUT_light = time.time() - start_time

# dark图片
# 有LUT
start_time = time.time()
result_LUT_dark = gamma_LUT(dark,gamma_dark)
time_LUT_dark = time.time() - start_time
# 无LUT
start_time = time.time()
result_no_LUT_dark = gamma_no_LUT(dark,gamma_dark)
time_no_LUT_dark = time.time() - start_time


# 显示结果
print("light_LUT 's time :",time_LUT_light)
print("light_no_LUT 's time :",time_no_LUT_light)
cv2.imshow("light",light)
cv2.imshow("light_LUT",result_LUT_light)
cv2.imshow("light_no_LUT",result_no_LUT_light)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("dark_LUT 's time :",time_LUT_dark)
print("dark_no_LUT 's time :",time_no_LUT_dark)
cv2.imshow("dark",dark)
cv2.imshow("dark_LUT",result_LUT_dark)
cv2.imshow("dark_no_LUT",result_no_LUT_dark)
cv2.waitKey(0)
cv2.destroyAllWindows()
