import cv2
import numpy as np
import csv
import time
import os
def main(output_dir='', label_dir='', name=None, oder=[]):
    # 定义一个空列表存储n个误差
    errors = []
    # 定义三个空列表存储n个汉明距离、杰卡德相似系数和Dice相似系数
    hamming_distances = []
    jaccard_similarities = []
    dice_similarities = []
    error_dir = './errors/'
    if not oder.all() : 
        oder = len((os.listdir(label_dir))) +1 
        oder = np.arange(1,oder,1)
        # 用一个for循环遍历n张图像
        for i in oder:
            output = cv2.imread(output_dir+f'{i}.tif', cv2.IMREAD_UNCHANGED)
            label = cv2.imread(label_dir+f'{i}.tif', cv2.IMREAD_UNCHANGED)
            output = output / 255.0
            label = label / 255.0
            output = output.astype(int)
            label = label.astype(int)
            # 调用error函数计算两个图像之间的误差，并添加到列表中
            errors.append(error(output, label))
            # 调用三个函数，计算两个图像之间的三个参数，并添加到对应的列表中
            hamming_distances.append(hamming_distance(output, label))
            jaccard_similarities.append(jaccard_similarity(output, label))
            dice_similarities.append(dice_similarity(output, label))
    else:
        for i in oder:
            filename = f'{i}.tif'
            output = cv2.imread(output_dir+filename, cv2.IMREAD_UNCHANGED)
            label = cv2.imread(label_dir+filename, cv2.IMREAD_UNCHANGED)
            output = output / 255.0
            label = label / 255.0
            output = output.astype(int)
            label = label.astype(int)
            # 调用error函数计算两个图像之间的误差，并添加到列表中
            errors.append(error(output, label))
            # 调用三个函数，计算两个图像之间的三个参数，并添加到对应的列表中
            hamming_distances.append(hamming_distance(output, label))
            jaccard_similarities.append(jaccard_similarity(output, label))
            dice_similarities.append(dice_similarity(output, label))

    # 计算100个误差的平均值，忽略-1的值
    mean_error = np.mean([e for e in errors if e != -1])
    # 计算三个参数的平均值，忽略-1的值，并打印出来
    mean_hamming_distance = np.mean([h for h in hamming_distances if h != -1])
    mean_jaccard_similarity = np.mean([j for j in jaccard_similarities if j != -1])
    mean_dice_similarity = np.mean([d for d in dice_similarities if d != -1])
    print(f'平均汉明距离为：{mean_hamming_distance:.4f}')
    print(f'平均杰卡德相似系数为：{mean_jaccard_similarity:.4f}')
    print(f'平均Dice相似系数为：{mean_dice_similarity:.4f}')
    # 打印平均误差
    print(f'平均误差为：{mean_error:.4f}%')

    # 将100个误差写入csv文件，假设文件名为errors.csv，位于当前目录下
    timestamp = time.strftime('%Y%m%d%H%M')
    if name == None:
        filename = 'errors_'+ timestamp +'.csv'
        name = error_dir + filename
    else: name = name
    with open(name, 'w') as f:
        # 创建一个csv写入对象
        writer = csv.writer(f)
        # 写入一行表头，表示图像编号和误差
        writer.writerow(['Image', 'Error', 'Hamming Distance', 'Jaccard Similarity', 'Dice Similarity'])
        writer.writerow(['Mean', mean_error, mean_hamming_distance, mean_jaccard_similarity, mean_dice_similarity])
        # 用一个for循环遍历n个误差，并写入一行数据，表示第i张图像的误差
        for i, (o,e,h,j,d) in enumerate(zip(oder, errors,hamming_distances, jaccard_similarities, dice_similarities)):
            writer.writerow([o, e,h,j,d])
# 定义一个函数计算两个图片矩阵的误差
def error(img1, img2):
    # 将图片矩阵转换为一维向量
    vec1 = img1.flatten()
    vec2 = img2.flatten()
    # 计算两个向量的差值的二范数的绝对值
    diff = np.linalg.norm(vec1 - vec2)
    # 计算真值向量的二范数
    norm = np.linalg.norm(vec2)
    # 返回误差，即差值除以真值
        # 判断真值向量是否为0或接近0
    if norm < 1e-6:
        # 返回一个特殊的值，表示无法计算误差
        return -1
    else:
        # 返回误差，即差值除以真值乘以100
        return diff / norm * 100
# 定义一个函数计算两个二值化图片矩阵的汉明距离
def hamming_distance(img1, img2):
    # 将图片矩阵转换为一维向量
    vec1 = img1.flatten()
    vec2 = img2.flatten()
    # 计算两个向量的异或，然后统计非零元素的个数
    diff = np.count_nonzero(vec1 ^ vec2)
    # 返回汉明距离
    return diff

# 定义一个函数计算两个二值化图片矩阵的杰卡德相似系数
def jaccard_similarity(img1, img2):
    # 将图片矩阵转换为一维向量
    vec1 = img1.flatten()
    vec2 = img2.flatten()
    # 计算两个向量的逻辑与，然后统计非零元素的个数（重叠区域）
    intersection = np.count_nonzero(vec1 & vec2)
    # 计算两个向量的逻辑或，然后统计非零元素的个数（总区域）
    union = np.count_nonzero(vec1 | vec2)
    # 返回杰卡德相似系数，即重叠区域除以总区域
        # 判断总区域是否为0或接近0
    if union < 1e-6:
        # 返回一个特殊的值，表示无法计算相似系数
        return -1
    else:
        # 返回杰卡德相似系数
        return intersection / union

# 定义一个函数计算两个二值化图片矩阵的Dice相似系数
def dice_similarity(img1, img2):
    # 将图片矩阵转换为一维向量
    vec1 = img1.flatten()
    vec2 = img2.flatten()
    # 计算两个向量的逻辑与，然后统计非零元素的个数（重叠区域）
    intersection = np.count_nonzero(vec1 & vec2)
    # 计算两个向量各自的非零元素的个数之和（平均区域）
    sum = np.count_nonzero(vec1) + np.count_nonzero(vec2)
    # 返回Dice相似系数，即重叠区域乘以2除以平均区域
        # 判断平均区域是否为0或接近0
    if sum < 1e-6:
        # 返回一个特殊的值，表示无法计算相似系数
        return -1
    else:
        # 返回Dice相似系数
        return (2 * intersection) / sum

    
    
# import cv2
# import numpy as np
# import csv
# import time
# import os
# # 定义一个函数计算两个图片矩阵的误差
# def error(img1, img2):
#     # 将图片矩阵转换为一维向量
#     vec1 = img1.flatten()
#     vec2 = img2.flatten()
#     # 计算两个向量的差值的二范数的绝对值
#     diff = np.linalg.norm(vec1 - vec2)
#     # 计算真值向量的二范数
#     norm = np.linalg.norm(vec2)
#     # 返回误差，即差值除以真值
#         # 判断真值向量是否为0或接近0
#     if norm < 1e-6:
#         # 返回一个特殊的值，表示无法计算误差
#         return -1
#     else:
#         # 返回误差，即差值除以真值乘以100
#         return diff / norm * 100
# # 定义一个函数计算两个二值化图片矩阵的汉明距离
# def hamming_distance(img1, img2):
#     # 将图片矩阵转换为一维向量
#     vec1 = img1.flatten()
#     vec2 = img2.flatten()
#     # 计算两个向量的异或，然后统计非零元素的个数
#     diff = np.count_nonzero(vec1 ^ vec2)
#     # 返回汉明距离
#     return diff

# # 定义一个函数计算两个二值化图片矩阵的杰卡德相似系数
# def jaccard_similarity(img1, img2):
#     # 将图片矩阵转换为一维向量
#     vec1 = img1.flatten()
#     vec2 = img2.flatten()
#     # 计算两个向量的逻辑与，然后统计非零元素的个数（重叠区域）
#     intersection = np.count_nonzero(vec1 & vec2)
#     # 计算两个向量的逻辑或，然后统计非零元素的个数（总区域）
#     union = np.count_nonzero(vec1 | vec2)
#     # 返回杰卡德相似系数，即重叠区域除以总区域
#         # 判断总区域是否为0或接近0
#     if union < 1e-6:
#         # 返回一个特殊的值，表示无法计算相似系数
#         return -1
#     else:
#         # 返回杰卡德相似系数
#         return intersection / union

# # 定义一个函数计算两个二值化图片矩阵的Dice相似系数
# def dice_similarity(img1, img2):
#     # 将图片矩阵转换为一维向量
#     vec1 = img1.flatten()
#     vec2 = img2.flatten()
#     # 计算两个向量的逻辑与，然后统计非零元素的个数（重叠区域）
#     intersection = np.count_nonzero(vec1 & vec2)
#     # 计算两个向量各自的非零元素的个数之和（平均区域）
#     sum = np.count_nonzero(vec1) + np.count_nonzero(vec2)
#     # 返回Dice相似系数，即重叠区域乘以2除以平均区域
#         # 判断平均区域是否为0或接近0
#     if sum < 1e-6:
#         # 返回一个特殊的值，表示无法计算相似系数
#         return -1
#     else:
#         # 返回Dice相似系数
#         return (2 * intersection) / sum
    
# def main(output_dir, label_dir):
#     # 定义一个空列表存储100个误差
#     errors = []
#     # 定义三个空列表存储100个汉明距离、杰卡德相似系数和Dice相似系数
#     hamming_distances = []
#     jaccard_similarities = []
#     dice_similarities = []
#     n = len((os.listdir(output_dir)))
            
#     # 用一个for循环遍历n张图像
#     for i in range(1, n+1):
#         output = cv2.imread(output_dir+f'{i}.tif', cv2.IMREAD_UNCHANGED)
#         label = cv2.imread(label_dir+f'{i}.tif', cv2.IMREAD_UNCHANGED)
#         output = output / 255.0
#         label = label / 255.0
#         output = output.astype(int)
#         label = label.astype(int)
#         # 调用error函数计算两个图像之间的误差，并添加到列表中
#         errors.append(error(output, label))
#         # 调用三个函数，计算两个图像之间的三个参数，并添加到对应的列表中
#         hamming_distances.append(hamming_distance(output, label))
#         jaccard_similarities.append(jaccard_similarity(output, label))
#         dice_similarities.append(dice_similarity(output, label))
#     # 计算100个误差的平均值，忽略-1的值
#     mean_error = np.mean([e for e in errors if e != -1])
#     # 计算三个参数的平均值，忽略-1的值，并打印出来
#     mean_hamming_distance = np.mean([h for h in hamming_distances if h != -1])
#     mean_jaccard_similarity = np.mean([j for j in jaccard_similarities if j != -1])
#     mean_dice_similarity = np.mean([d for d in dice_similarities if d != -1])
#     print(f'平均汉明距离为：{mean_hamming_distance:.4f}')
#     print(f'平均杰卡德相似系数为：{mean_jaccard_similarity:.4f}')
#     print(f'平均Dice相似系数为：{mean_dice_similarity:.4f}')
#     # 打印平均误差
#     print(f'平均误差为：{mean_error:.4f}%')

#     # 将100个误差写入csv文件，假设文件名为errors.csv，位于当前目录下
#     timestamp = time.strftime('%Y%m%d%H%M')
#     filename = 'errors_'+ timestamp +'.csv'
#     with open(filename, 'w') as f:
#         # 创建一个csv写入对象
#         writer = csv.writer(f)
#         # 写入一行表头，表示图像编号和误差
#         writer.writerow(['Image', 'Error', 'Hamming Distance', 'Jaccard Similarity', 'Dice Similarity'])
#         writer.writerow(['Mean', mean_error, mean_hamming_distance, mean_jaccard_similarity, mean_dice_similarity])
#         # 用一个for循环遍历n个误差，并写入一行数据，表示第i张图像的误差
#         for i, (e,h,j,d) in enumerate(zip(errors,hamming_distances, jaccard_similarities, dice_similarities)):
#             writer.writerow([i + 1, e,h,j,d])