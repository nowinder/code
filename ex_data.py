import cv2 
import numpy as np
import scipy.io

# 读取二十张灰度图像
for n in range(1,901):
	img_list = []
	for ch in range(1, 21):
		img = cv2.imread(f"/data/caizhizheng/data/data/CH{ch}/{n}.tif", 0) # 0或cv2.IMREAD_GRAYSCALE表示读取灰度图像
		img_list.append(img)

	# 扩充二十张灰度图像到32*32的尺寸，并用0像素值填充
	img_pad_list = []
	for img in img_list:
		img_pad = cv2.copyMakeBorder(img, 6,6,6,6,cv2.BORDER_CONSTANT, value= 0) 
		img_pad_list.append(img_pad)

	# 将扩充后的灰度图像合并为一个三维数组
	img_merge = np.dstack(img_pad_list)

	# 保存合并后的图像
	scipy.io.savemat(f"/data/caizhizheng/data/ndata/data/{n}.mat", {"img_merge": img_merge})
	img_pad_list = []
	
	