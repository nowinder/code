from PIL import Image
import numpy as np

image1 = Image.open('/code/predict_data/denseunet_1.png')
# image2 = Image.open('/data/caizhizheng/data/label/label/1.tif')

matrix1 = np.array(image1)
#matrix2 = np.array(image2)

print(matrix1)
#print(matrix2)