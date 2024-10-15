# 任务1
# 写一个图片放大的程序，可以将图片放大指定倍数
# 至少包括最近邻、双线性插值两种方法（需自己从底层实现，不可调用函数）
# 在2倍的放大倍数下（长宽各乘2），和OpenCV的resize函数来比较放大效果和计算效率
# 用https://unlimited.waifu2x.net/ 比较一下和深度学习图片放大算法，放大2倍的效果
# 以上比较，至少在test1.jpg和test2.jpg上进行，并和原图test1_origin.jpg和test2_ origin.jpg进行比较

import cv2
import self_algrithm
import time

# 加载原图像
image1 = cv2.imread("test1.jpg", 1)
image2 = cv2.imread("test2.jpg", 1)
# org_h1, org_w1 = image1.shape[:2]
# org_h2, org_w2 = image2.shape[:2]

# 缩放因子scale
scale = 2

# ------------------------------------处理第一张图------------------------------------------------
print("Processing image: test1.jpg")

# 生成我实现的图像
start_time = time.time()
my1 = self_algrithm.resize(image1, scale)
my_time_1 = time.time() - start_time

# 保存结果
cv2.imwrite('my_new1.jpg', my1)

# 生成运用cv2库里的resize()的图像
# 函数原型：cv2.resize(src, dsize, dst=None, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
# cv2.INTER_NEAREST：最近邻插值
start_time = time.time()
cv1_nearest = cv2.resize(image1, None, dst=None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
cv1_time_nearest = time.time() - start_time
# cv2.INTER_LINEAR：双线性插值
start_time = time.time()
cv1_inter_linear = cv2.resize(image1, None, dst=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
cv1_time_linear = time.time() - start_time

# 打印处理时间
print(f"My Nearest Neighbor and Bilinear resize time (image1): {my_time_1:.4f} seconds")
print(f"OpenCV Nearest Neighbor resize time (image1): {cv1_time_nearest:.4f} seconds")
print(f"OpenCV Bilinear resize time (image1): {cv1_time_linear:.4f} seconds")

# 比较效果
cv2.imshow("Original image1", image1)
cv2.imshow("My image1", my1)
cv2.imshow("cv2_NEAREST 's image1", cv1_nearest)
cv2.imshow("cv2_INTER_LINEAR 's image1", cv1_inter_linear)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------------处理第二张图------------------------------------------------
print("Processing image: test2.jpg")

# 生成我实现的图像
start_time = time.time()
my2 = self_algrithm.resize(image2, scale)
my_time_2 = time.time() - start_time

# 保存结果
cv2.imwrite('my_new2.jpg', my2)

# 生成运用cv2库里的resize()的图像
# 函数原型：cv2.resize(src, dsize, dst=None, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
# cv2.INTER_NEAREST：最近邻插值
start_time = time.time()
cv2_nearest = cv2.resize(image2, None, dst=None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
cv2_time_nearest = time.time() - start_time
# cv2.INTER_LINEAR：双线性插值
start_time = time.time()
cv2_inter_linear = cv2.resize(image2, None, dst=None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
cv2_time_linear = time.time() - start_time

# 打印处理时间
print(f"My Nearest Neighbor and Bilinear resize time (image2): {my_time_2:.4f} seconds")
print(f"OpenCV Nearest Neighbor resize time (image2): {cv2_time_nearest:.4f} seconds")
print(f"OpenCV Bilinear resize time (image2): {cv2_time_linear:.4f} seconds")

# 比较效果
cv2.imshow("Original image2", image2)
cv2.imshow("My image2", my2)
cv2.imshow("cv2_NEAREST 's image2", cv2_nearest)
cv2.imshow("cv2_INTER_LINEAR 's image2", cv2_inter_linear)
cv2.waitKey(0)
cv2.destroyAllWindows()
