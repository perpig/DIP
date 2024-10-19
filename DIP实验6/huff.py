# 编程对图片bridge.bmp和web.bmp进行压缩
# 1.	采用哈夫曼编码，实现压缩和解压缩
# 2.	采用无损预测编码，并对误差进行哈夫曼编码，实现压缩和解压缩（该小题选做）
# 3.	用平均均方误差的平方根（如下），对解压缩后的图像和原图进行比较，并计算压缩比

import cv2
import numpy as np
from collections import Counter
import heapq # 堆

# -----------------------------------计算频率并构建哈夫曼树-------------------------------
# 定义哈夫曼树节点
class HuffmanNode:
    def __init__(self, value, freq):
        self.value = value # 值
        self.freq = freq # 频率
        self.left = None # 左子树
        self.right = None # 右子树

    # 重载<
    # 比较两个节点，频率较小的优先
    # 在heap形成最小堆的时候用到  否则heapq不知道哪个大
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(frequencies): # 构建huffman树
    # 先创建堆
    heap = [HuffmanNode(value, freq) for value, freq in frequencies.items()]
    # 转化为最小堆
    heapq.heapify(heap)

    # 构建huffman树
    while len(heap) > 1:  # 剩一个的时候就是根结点
        # heapq.heappop(heap)用于从最小堆中弹出并返回堆顶的元素。
        # 哈夫曼树的构建规则之一：每次合并两个频率最小的节点，生成一个新的节点，频率是这两个节点频率之和。
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heapq.heappush(heap,merged)
        heapq.heapify(heap)

    return heap[0]

# 生成哈夫曼编码
def generate_codes(node, code="", codes={}):
    if node:
        if node.value is not None:
            codes[node.value] = code

        generate_codes(node.left, code + "0", codes)
        generate_codes(node.right, code + "1", codes)
    return codes


# -----------------------------压缩图片---------------------------
def compress_image(img, codes):
    # 连接所有编码组成原始数据
    compress_data = ''.join([codes[pixel] for pixel in img.flatten()])

    # 填充
    padding = 8 - len(compress_data) % 8
    compress_data += '0' * padding

    data = bytearray()
    for i in range(0, len(compress_data), 8):
        byte = compress_data[i:i + 8]
        data.append(int(byte, 2))

    return data, padding


def calculate_frequency(img):  # 计算频率
    # Counter(...)：ollections 模块中的一个类，专门用于统计可哈希对象的出现次数。
    return Counter(img.flatten())


def huffman_compress(img):   # 对图片进行压缩
    frequencies = calculate_frequency(img)
    huffman_tree = build_huffman_tree(frequencies)
    huffman_codes = generate_codes(huffman_tree)
    compress_data, padding = compress_image(img, huffman_codes)
    return compress_data, huffman_codes, padding

# --------------------------解压缩图片-----------------------
def huffman_decompress(compressed_data, codes, padding, original_shape):
    # 反向编码表
    reverse_codes = {code: pixel for pixel, code in codes.items()}
    # 还原原数据
    original_data = ''.join(f'{byte:08b}' for byte in compressed_data)
    original_data = original_data[:-padding]

    decoded_pixels = []
    code = ''
    for bit in original_data:
        code += bit
        if code in reverse_codes:
            decoded_pixels.append(reverse_codes[code])
            code = ''

    img = np.array(decoded_pixels).reshape(original_shape)
    return img


# 测试图片
def test_photo(filename):
    # 读取
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # 压缩图像
    compressed_data, codes, padding = huffman_compress(image)

    # 解压图像
    decompressed_image = huffman_decompress(compressed_data, codes, padding, image.shape)

    # 显示解压后的图像
    cv2.imshow("Original Image", image)
    cv2.imshow("Decompressed Image", decompressed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    test_photo('bridge.bmp')
    # test_photo('web.bmp')
