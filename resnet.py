# 用自带的来改似乎有许多问题
import os
import shutil
from unittest import result
import skimage.io as io
import skimage.transform as trans
from keras.optimizer_v2.adam import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras 
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.saving.utils_v1.mode_keys import is_train
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
import imageio.v2 as imageio
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import models
from keras import layers
# from keras.layers import BatchNormalization, Conv2D, Activation, Dropout, AveragePooling2D, concatenate, \
    # GlobalAveragePooling2D, MaxPooling2D, Dense, Input
from keras.regularizers import l2
import keras.backend as K
import evalu
import time
import zipfile
from keras.applications import resnet
from keras.regularizers import l2
import random

def getimg(dir, j):
    input_dir = dir
    img_list = []
    for i in range(1,1201):
        filename = f'{i+j}.tif'
        img_path = os.path.join(input_dir,filename)
        img = imageio.imread(img_path)
        img = img.astype(np.float32)
        img = img / 255.0
        img_list.append(img)
        # print(i)
    return img_list

def onehot(img):
    img = tf.cast(img, tf.int32)
    img_one_hot = tf.one_hot(img, depth=2, on_value=1.0, off_value=0.0, axis=-1)
    return img_one_hot

# def add_channel(img): 
#     img = tf.expand_dims(img, axis=-1) 
#     return img
# Draw loss curve
def plot_history(history, result_dir, prefix):
    """
    将训练与验证的accuracy与loss画出来
    """
    plt.plot(history.history['accuracy'], marker='.')
    plt.plot(history.history['val_accuracy'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.yscale("log")
    plt.legend(['acc'], loc='upper right')
	# plt.show()
    plt.savefig(result_dir + 'ace' + prefix + '.png')
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale("log")
    plt.grid()
    plt.legend(['loss'], loc='upper right')
    # plt.show()
    plt.savefig(result_dir + 'loss'+ prefix +'.png')
    plt.close()

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=1000,
  decay_rate=1,
  staircase=False)
        
def make_zip(source_dir, output_name):
    zipf = zipfile.ZipFile(output_name, 'w')
    prelen = len(os.path.dirname(source_dir))
    for parent, _, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[prelen:].strip(os.path.sep)     #相对路径
            zipf.write(pathfile, arcname)
        zipf.close()

def train(model, model_savename, train_dataset, validation_dataset):
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    # reduce_lr = LearningRateScheduler(scheduler)
    # reduce_lr = LearningRateScheduler(lschedule)
    model_checkpoint = ModelCheckpoint('./save_model/'+model_savename, monitor='loss', verbose=1,
                                       save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
    Board = tf.keras.callbacks.TensorBoard(log_dir="/output/logs")

    history = model.fit(train_dataset.repeat(),
                                  steps_per_epoch=100,
                                  epochs=1000,
                                  validation_data=validation_dataset,
                                #   validation_steps=50,
                                  callbacks=[model_checkpoint,
                                             early_stop,Board
                                             # Board
                                             ])
    plot_history(history, './loss_picture/', '_resnet')
    return model

        
BATCH_SIZE=32

if __name__ == '__main__':
    save_path = '/data/caizhizheng/2D/newnet/'
    label_path = '/data/caizhizheng/2D/v2/label/label'
    # data_path = save_path + 'data/'
    output_path = save_path + 'predict/'
    # output_path = '/data/caizhizheng/2D/v2/exped_output/'
    label_path = '/data/caizhizheng/2D/v2/label/label/'
    img_path = '/data/caizhizheng/2D/v2/data1/data/'
    img1_list = getimg(img_path, j=0)
    img_path = '/data/caizhizheng/2D/v2/data2/data/'
    img2_list = getimg(img_path, j=900)
    label_list = getimg(label_path,j=0)
    img_list = np.stack((img1_list,img2_list),axis = -1)
    image_dataset = tf.data.Dataset.from_tensor_slices(img_list)
    label_dataset = tf.data.Dataset.from_tensor_slices(label_list)
    label_dataset = label_dataset.map(onehot)
    train_dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    train_dataset = train_dataset.cache().shuffle(train_dataset.cardinality())

    validation_dataset = train_dataset.take(50).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    train_dataset = train_dataset.skip(50).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # 加载 ResNet50 模型，去掉最后一层分类层，指定输入尺寸和通道数
    base_model = resnet.ResNet50(weights=None, include_top=False, input_shape=(128, 128, 2))
    # 修改第一层卷积的输入通道数
    # base_model.layers[1] = keras.layers.Conv2D(2, (7, 7), strides=(2, 2), padding='same', name='conv1_pad')
    # 在模型的最后添加一个上采样层，将特征图放大到 128×128
    x = layers.UpSampling2D(size=(32, 32))(base_model.output)
    # 在模型的最后添加一个卷积层，将特征图转换为输出图像，激活函数为 tanh
    output = layers.Conv2D(2, (3, 3), padding='same', activation='softmax')(x)
    # 构建图像图像映射的网络，输入和输出都是 128×128×2 的图像
    model = models.Model(inputs=base_model.input, outputs=output)
    model.summary()
    model_savename = 'resnet.keras'
    train = train(model, model_savename, train_dataset, validation_dataset)
    
    image_dataset = image_dataset.batch(BATCH_SIZE)
    mask = model.predict(image_dataset)
    mask = tf.argmax(mask, axis=-1)
    mask = tf.keras.backend.eval(mask)
    mask = (mask * 255).astype(np.uint8)
    os.makedirs(output_path, exist_ok=True)
    for j in range(0, 1200):
        cv2.imwrite(output_path+ '%d.tif' %(j+1), mask[j])

    # name = 'error2' + '_postCrV' + '.csv'
    name = 'error' + '.csv'
    name = os.path.join(save_path, name)
    evalu.main(output_dir=output_path, label_dir=label_path, name=name, oder=np.array(0))

    # output_dir = '/data/caizhizheng/2D/validataset/predict2/'
    # output_name = '/data/caizhizheng/2D/validataset/' + 'vali-filterd2.zip'
    # output_name = output_path + 'predict.zip'
    # make_zip(source_dir=output_path, output_name=output_name)
    # print("zip OK")