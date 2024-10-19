# 编程对图片bridge.bmp和web.bmp进行压缩
# 1.	采用哈夫曼编码，实现压缩和解压缩
import cv2
import numpy as np
from collections import Counter
import heapq  # 堆

# -----------------------------------计算频率并构建哈夫曼树-------------------------------
# 定义哈夫曼树节点
class HuffmanNode:
    def __init__(self, value, freq):
        self.value = value # 值
        self.freq = freq  # 频率
        self.left = None  # 左子树
        self.right = None # 右子树

    # 重载小于 (<) 运算符。
    # 比较两个节点，频率较小的优先
    # 在heap形成最小堆的时候用到  否则heapq不知道哪个大
    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(frequencies):
# 构建哈夫曼树
    # frequencies是一个字典（dict）
    # 形式：{value1: freq1, value2: freq2, ...}，其中value是要编码的元素（例如字符），freq是该元素出现的频率。
    # for value, freq in frequencies.items()       frequencies.items()迭代器
    heap = [HuffmanNode(value, freq) for value, freq in frequencies.items()]
    # 将这个列表转换为最小堆。
    heapq.heapify(heap)

    # 由heap这个最小堆序列 构建huff树
    while len(heap) > 1:
        # heapq.heappop(heap)用于从最小堆中弹出并返回堆顶的元素。（这里是根结点）
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        # 这个节点没有具体的值
        merged = HuffmanNode(None, node1.freq + node2.freq)
        # 哈夫曼树的构建规则之一：每次合并两个频率最小的节点，生成一个新的节点，频率是这两个节点频率之和。
        merged.left = node1
        merged.right = node2
        # 把这个新节点放在堆顶
        heapq.heappush(heap, merged)

    # 返回根结点
    return heap[0]

# 生成哈夫曼编码
def generate_codes(node, code="", codes={}):
    # 不为空
    if node:
        if node.value is not None:
            # 记录这个有值节点的编码
            codes[node.value] = code
        # 递归实现
        generate_codes(node.left, code + "0", codes)
        generate_codes(node.right, code + "1", codes)
    return codes


# ------------------------------------------压缩图片-------------------------------------------
def compress_image(img, codes):
# 把图像和弄好的code编码和值对应的hash 转为数据（二进制字节数组）
    # 连接所有编码组成原始数据
    # 将图片像素转为哈夫曼编码
    # join()字符串方法 元素连接成字符串   ''是空字符串
    # flatten 二维数组展平为一维
    compressed_data = ''.join([codes[pixel] for pixel in img.flatten()])

    # 将位流转换为字节 字节必须是8的倍数
    # 计算要填充多少个0
    padding = 8 - len(compressed_data) % 8
    # 用'0'填充 变成8的倍数
    compressed_data += '0' * padding

    # 保存压缩信息：
    # bytearray()：创建一个可变的字节数组，用于存储最终的压缩数据。
    b = bytearray()
    for i in range(0, len(compressed_data), 8):
        # 提取这一个字节的字符串内容
        byte = compressed_data[i:i + 8]
        # 从字符串转为二进制
        # int(x, base)   x：要转换的字符串  base：表示数字的进制，2 表示二进制
        b.append(int(byte, 2))

    # 返回二进制字节数组 填充0个数
    return b, padding


# 计算频率
def calculate_frequency(img):
    # Counter(...)：ollections 模块中的一个类，专门用于统计可哈希对象的出现次数。
    # {像素值：出现频率}
    return Counter(img.flatten())


# 对图片进行压缩
def huffman_compress(img):
    # 计算频率(用来build huffman)
    frequencies = calculate_frequency(img)

    # 构建哈夫曼树 返回的根结点
    huffman_tree = build_huffman_tree(frequencies)

    # 用构建好的树生成哈夫曼编码(递归实现)   传入根结点
    codes = generate_codes(huffman_tree)
    # codes[像素值] = 编码

    # 压缩图片成为二进制字节数组
    compressed_data, padding = compress_image(img, codes)

    # # 计算原始图像大小（字节数）
    # original_size = img.size  # 像素总数（灰度图每个像素1字节）
    #
    # # 压缩后数据大小（字节数）
    # compressed_size = len(compressed_data)  # 字节数组的长度
    #
    # # 计算压缩比
    # compression_ratio = original_size / compressed_size
    #
    # print(f"Original size: {original_size} bytes")
    # print(f"Compressed size: {compressed_size} bytes")
    # print(f"Compression Ratio: {compression_ratio:.2f}")

    return compressed_data, codes, padding, huffman_tree


# --------------------------------------------解压缩图片--------------------------------------
def huffman_decompress(compressed_data, codes, padding, original_shape):
    # 将字节流转换为位流  :08b是bytes格式化为8位二进制前面补0
    # 压缩时候每个字节转成数字二进制的时候 前面的0会省去
    bit_string = ''.join(f'{byte:08b}' for byte in compressed_data)

    # 移除填充位
    bit_string = bit_string[:-padding]

    # 生成哈夫曼反向编码表
    reverse_codes = {code: pixel for pixel, code in codes.items()}

    # 解码位流
    decoded_pixels = []
    code = ""
    for bit in bit_string:
        code += bit
        if code in reverse_codes:
        # 当找到一个huffman编码的时候说明这是一个像素了
            decoded_pixels.append(reverse_codes[code])
            code = ""
            # 检查解码后的像素长度
            if len(decoded_pixels) >= original_shape[0] * original_shape[1]:
                break

    # 这时候decoded_pixels 已经是一个图像了 只不过是一维的

    # 打印解码后像素的数量
    print(f"Decoded pixels length: {len(decoded_pixels)}")

    # 还原图片
    img = np.array(decoded_pixels).reshape(original_shape)
    return img

# ---------------------------------------------测试图片-----------------------------------
def test_photo(filename):
    print(f"------------------开始测试图像:{filename}-----------------")
    # 读取
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    print(f"开始压缩图像:{filename}")
    # 压缩图像
    compressed_data, codes, padding, tree = huffman_compress(image)
    print(f"压缩图像完毕:{filename}")

    print(f"开始解压图像数据:{filename}")
    # 解压图像
    decompressed_image = huffman_decompress(compressed_data, codes, padding, image.shape)
    print(f"解压图像数据完毕:{filename}")

    print(f"显示图像压缩解压前后效果:{filename}")
    # 显示解压后的图像
    cv2.imshow("Original Image", image)
    cv2.imshow("Decompressed Image", decompressed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    test_photo('bridge.bmp')
    # test_photo('web.bmp')
