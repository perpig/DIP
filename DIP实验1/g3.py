import cv2
import numpy as np

# 读取&转灰度
image = cv2.imread('letter.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用阈值，将图像转换为二值图像
_, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# -----------------------------------提取char轮廓----------------------------------------
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个空白图像来绘制轮廓（选择与原始图像相同大小）
contour_image = np.ones_like(image) * 255  # 白色背景图像

# 查看情况
cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)  # 用红色线条绘制轮廓，线条宽度为2
cv2.imshow('Contours', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------根据轮廓截取字符图像并存储----------------------------------

characters = {}
max_width = 0
max_height = 0
# 遍历每个轮廓并裁剪出对应的图像区域
for i, contour in enumerate(contours):
    # 获取轮廓的边界框
    x, y, w, h = cv2.boundingRect(contours[i])

    # 过滤掉太小的轮廓
    if w < 10 or h < 10:
        continue

    # 从原始图像中裁剪出轮廓区域
    cropped_image = image[y:y + h, x:x + w]

    # 显示裁剪后的图像
    cv2.imshow(f'Cropped Image {i + 1}', cropped_image)
    print("输入该字符是哪个？（输小写字母和阿拉伯数字） (回车跳过此字符): ")
    key = cv2.waitKey(0)  # 等待用户按键

    # 判断用户输入的键
    if key == 13:  # 如果按下回车键，跳过当前字符
        cv2.destroyWindow(f'Cropped Image {i + 1}')
        continue

    max_width = max(max_width, cropped_image.shape[1])  # 求最大宽度
    max_height = max(max_height, cropped_image.shape[0])  # 求最大高度
    # 将用户输入的键作为索引键存储裁剪的图像
    characters[chr(key)] = cropped_image
    cv2.destroyWindow(f'Cropped Image {i + 1}')

# 查看存储效果，显示A字母
cv2.imshow('A', characters['a'])
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------------生成新的图片-----------------------------------------
# 目标文字设置
id = input('输入学号：')  # 用户输入目标字符
name = input('输入名字拼音：')

# 设置字符间隔和行间距
char_spacing = 5  # 字符之间的间隔
line_spacing = 5  # 行间距

# 计算每个字符的宽度之和
def calculate_total_width(string):
    total_width = 0
    for char in string:
        if char in characters:
            total_width += characters[char].shape[1]  # 累加字符宽度
        else:
            print(char, '字符未找到')
    return total_width

# 计算新图像的总宽度和高度
total_width = max(calculate_total_width(id),calculate_total_width(name)) + (len(id) - 1) * char_spacing + char_spacing * 4
total_height = max_height * 2 + line_spacing * 2
print('total_width:', total_width)
print('total_height:', total_height)

# 选择原图中的一个区域作为背景颜色的参考
reference_region = image[0:10, 0:10]  # 例如选取原图的左上角10x10区域
background_color = np.mean(reference_region, axis=(0, 1)).astype(int)  # 计算平均颜色

# 确保背景颜色为uint8
background_color = np.clip(background_color, 0, 255).astype(np.uint8)

# 创建一个空白的背景图像（使用提取的背景颜色）
new_image = np.ones((total_height, total_width, 3), dtype=np.uint8) * background_color

# --------------------------------拼接第一行学号--------------------------------
current_x = char_spacing  # 从字符间隔开始
current_y = char_spacing  # 从字符间隔开始
for char in id:
    if char in characters:
        char_image = characters[char]  # 获取字符图像
        char_height = char_image.shape[0]
        char_width = char_image.shape[1]

        # 将字符图像粘贴到新图像的指定位置
        if char_width > (total_width - current_x):
            # If character width is too large, resize it
            char_image = cv2.resize(char_image, (total_width - current_x, char_height))

        new_image[current_y:current_y + char_height, current_x:current_x + char_width] = char_image
        current_x += char_width + char_spacing  # 更新x坐标
    else:
        print(char, '字符未找到')

# --------------------------------拼接第二行名字--------------------------------
current_x = char_spacing  # 重置x坐标
current_y += max_height + line_spacing  # 更新y坐标 加上行间距

for char in name:
    if char in characters:
        char_image = characters[char]  # 获取字符图像
        char_height = char_image.shape[0]
        char_width = char_image.shape[1]

        # 将字符图像粘贴到新图像的指定位置
        if char_width > (total_width - current_x):
            # If character width is too large, resize it
            char_image = cv2.resize(char_image, (total_width - current_x, char_height))

        new_image[current_y:current_y + char_height, current_x:current_x + char_width] = char_image
        current_x += char_width + char_spacing  # 更新x坐标
    else:
        print(char, '字符未找到')

# 显示拼接完成的图像
cv2.imshow('New Image', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
cv2.imwrite('output_image.png', new_image)
