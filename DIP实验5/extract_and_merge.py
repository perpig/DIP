# 任务3
# 编程将green图片中的人物抠出，并融合到tree图片中，力求融合结果自然

import cv2
import numpy as np

# 扣人物的函数
def extract_person(green_filename):
    # 提取照片
    image = cv2.imread(green_filename, cv2.IMREAD_COLOR)
    # 转化为hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义绿色的HSV阈值范围
    # H（Hue）色调：表示颜色的基本色调，范围为 0 到 180，绿色的色调大概在 35 到 85 之间。
    lower_green = np.array([35, 40, 40])
    # 这是定义绿色的最低 HSV 值。35表示色调的最小值，对应于绿色的下限。
    # 40表示饱和度的最小值，表示提取饱和度至少为40的绿色。40表示亮度的最小值，提取亮度至少为40的绿色部分。
    upper_green = np.array([85, 255, 255])
    # 这是定义绿色的最高HSV值。85表示色调的最大值，对应于绿色的上限。
    # 255表示饱和度的最大值，表示可以提取饱和度为255的绿色。255表示亮度的最大值，表示可以提取最亮的绿色。

    # 提取绿色区域
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 和原图取反 留下人物
    mask_person = cv2.bitwise_not(mask_green)

    # 按照mask在原图上把人物扣下来
    person = cv2.bitwise_and(image, image, mask=mask_person)

    return person, mask_green

# 把人加到背景上的函数
def merge_with_tree(background_filename, person, mask_green):
    # 提取
    background = cv2.imread(background_filename, cv2.IMREAD_COLOR)

    # 准备背景图，将背景区域提取出来，用来放置提取出来的人物
    rows1, cols1, _ = person.shape
    rows2, cols2, _ = background.shape
    # 计算要放置人物图像的起始行坐标
    y_start = rows2 - rows1

    # 提取一块person大小的区域出来
    roi = background[y_start:rows2, 100:100 + cols1]

    # 把这个区域原图上人物区域扣掉
    roi = cv2.bitwise_and(roi, roi, mask=mask_green)
    # 然后加上人物的图像
    roi = cv2.add(person, roi)

    # 把原图这个区域替换为我们加了人物的
    background[y_start:rows2, 100:100 + cols1] = roi
    return background


# 主函数
def main():
    green_image_path = 'green.png'
    tree_image_path = 'tree.jpg'

    # 提取人物
    person, mask_background = extract_person(green_image_path)

    # 将人物融合到树的背景上
    result = merge_with_tree(tree_image_path, person, mask_background)

    # 显示结果
    cv2.imshow("Merged Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
