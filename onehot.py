import tensorflow as tf
import imageio.v2 as imageio
import os
import numpy as np
input_dir = '/data/caizhizheng/2D/v2/label/label'
img_list = []
for i in range(1,901):
    filename = f'{i}.tif'
    img_path = os.path.join(input_dir,filename)
    img = imageio.imread(img_path)
    img = img.astype(np.float32)
    img = img / 255.0
    img_list.append(img)
label_dataset = tf.data.Dataset.from_tensor_slices(label_paths)
    # img_one_hot = tf.one_hot(img, depth=2, on_value=255.0, off_value=0.0, axis=-1)