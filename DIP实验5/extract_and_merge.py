# 任务3
# 编程将green图片中的人物抠出，并融合到tree图片中，力求融合结果自然

import cv2
import numpy as np

def extract_person():
    return

def merge_with_tree():
    return

# 主函数
def main():
    green_image_path = 'green.png'
    tree_image_path = 'tree.jpg'

    # 提取人物

    # 将人物融合到树的背景上
    result = merge_with_tree()

    # 显示结果
    cv2.imshow("Merged Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
