# 任务2
# 自己实现直方图均衡化，并和OpenCV中对应函数进行效果对比，绘出变换前后的直方图。至少在如下图像上测试：school, baby, hill

import  cv2
import numpy as np
# import matplotlib.pyplot as plt

# 自实现原理和过程：
# 1.计算累积分布函数（CDF）：
# CDF 是对频率分布函数（PDF）的积分，它表示每个像素值在原始图像中出现的概率。CDF 可以通过对 PDF 进行累加计算得到
# 2.构建新的灰度级映射函数：
# Sk = (L - 1) * C(rk)
# rk为某个像素值，Sk是新的灰度值，范围仍在 [0,L−1] 之间
# 该公式表示累积分布函数的归一化映射，将原始图像的灰度值范围拉伸至整个灰度级范围。

def my_equalizeHist(src):
    # 第一步 计算直方图
    # flatten() 方法将其展平为一维数组，bins：表示直方图的区间数，range：定义直方图的范围
    # 返回值：hist--每个元素表示对应区间中的像素数量；bins--长度为（bins + 1），表示每个区间的边界值
    hist, bins = np.histogram(src.flatten(), 256, [0, 256])

    # 第二步 计算累积分布函数 (CDF)
    cdf = hist.cumsum()  # 累加和

    # 第三步：将 CDF 归一化到范围 [0, 255]
    cdf_min = cdf.min()  # 找到 CDF 中的最小非零值，避免除以零的情况
    cdf_max = cdf.max()
    cdf_normalized = ((cdf - cdf_min) * 255 / (cdf_max - cdf_min)).astype('uint8')  # 归一化，并确保灰度级是整型
    # cdf_normalized = (cdf * hist.max() / cdf.max())   # 对 CDF 进行归一化,

    return cdf_normalized[src]  # 返回归一化后的cdf图像

# def plot_histogram(image, title):
#     # 计算直方图并绘制
#     plt.hist(image.flatten(), 256, [0, 256], color='r')
#     plt.title(title)
#     plt.xlabel('Pixel Intensity')
#     plt.ylabel('Frequency')
#     plt.show()

# 读取图片
hill = cv2.imread("hill.jpg",0) # cv2.IMREAD_GRAYSCALE
baby = cv2.imread("baby.png", 0)
school = cv2.imread("school.png", 0)

# 用cv2处理
result_cv_hill = cv2.equalizeHist(hill)
result_cv_baby = cv2.equalizeHist(baby)
result_cv_school = cv2.equalizeHist(school)

# 用我自己实现的函数处理
result_my_hill = my_equalizeHist(hill)
result_my_baby = my_equalizeHist(baby)
result_my_school = my_equalizeHist(school)

# 显示结果/结果比较
# hill
cv2.imshow("original image of hill",hill)
cv2.imshow("cv2's result of hill",result_cv_hill)
cv2.imshow("my result of hill",result_my_hill)
cv2.waitKey(0)
cv2.destroyAllWindows()
# baby
cv2.imshow("original image of baby",baby)
cv2.imshow("cv2's result of baby",result_cv_baby)
cv2.imshow("my result of baby",result_my_baby)
cv2.waitKey(0)
cv2.destroyAllWindows()
# school
cv2.imshow("original image of school",school)
cv2.imshow("cv2's result of school",result_cv_school)
cv2.imshow("my result of school",result_my_school)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # 绘制原始图像和均衡化后图像的直方图
# for name, original, cv_result, my_result in zip(
#     ["hill", "baby", "school"],
#     [hill, baby, school],
#     [result_cv_hill, result_cv_baby, result_cv_school],
#     [result_my_hill, result_my_baby, result_my_school]
# ):
#     # 显示原始和均衡化后图像
#     cv2.imshow(f"Original Image ({name})", original)
#     cv2.imshow(f"OpenCV Equalized Image ({name})", cv_result)
#     cv2.imshow(f"My Equalized Image ({name})", my_result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#     # 绘制直方图
#     plot_histogram(original, f'Original Histogram ({name})')
#     plot_histogram(cv_result, f'OpenCV Equalized Histogram ({name})')
#     plot_histogram(my_result, f'My Equalized Histogram ({name})')

