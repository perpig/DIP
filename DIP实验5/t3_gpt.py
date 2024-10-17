# 任务3
# 编程将green图片中的人物抠出，并融合到tree图片中，力求融合结果自然

import cv2
import numpy as np

def extract_person(green_image_path):
    # 读取 green 图片
    green_image = cv2.imread(green_image_path)

    # 转换为 HSV 色彩空间
    hsv = cv2.cvtColor(green_image, cv2.COLOR_BGR2HSV)

    # 定义绿色的范围（可能需要根据实际情况调整）
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    # 创建掩码，提取绿色区域
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 反转掩码，保留人物
    mask_inv = cv2.bitwise_not(mask)

    # 使用掩码提取人物
    person = cv2.bitwise_and(green_image, green_image, mask=mask_inv)

    return person, mask_inv

def merge_with_tree(tree_image_path, person, mask_inv):
    # 读取 tree 图片
    tree_image = cv2.imread(tree_image_path)

    # 确保 tree 图片与人物抠图的大小一致
    height, width = person.shape[:2]
    tree_image_resized = cv2.resize(tree_image, (width, height))

    # 使用掩码将人物和树合并
    # 取树的前景部分
    tree_foreground = cv2.bitwise_and(tree_image_resized, tree_image_resized, mask=mask_inv)

    # 合并树的前景和人物
    result = cv2.add(tree_foreground, person)

    return result

# 主函数
def main():
    green_image_path = 'green.png'  # 替换为你的文件路径
    tree_image_path = 'tree.jpg'      # 替换为你的文件路径

    # 提取人物
    person, mask_inv = extract_person(green_image_path)

    # 将人物融合到树的背景上
    result = merge_with_tree(tree_image_path, person, mask_inv)

    # 显示结果
    cv2.imshow("Merged Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
