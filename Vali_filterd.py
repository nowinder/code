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
from keras import backend as keras
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
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Activation, Dropout, AveragePooling2D, concatenate, \
    GlobalAveragePooling2D, MaxPooling2D, Dense, Input
from keras.regularizers import l2
import keras.backend as K
import evalu
import time
import zipfile

def getimg(dir, oder, j):
    input_dir = dir
    img_list = []
    for i in oder:
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

def add_channel(img): 
    img = tf.expand_dims(img, axis=-1) 
    return img
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


def Conv_Block(input_tensor, filters, bottleneck=False, weight_decay=1e-4):
    """    封装卷积层

    :param input_tensor: 输入张量
    :param filters: 卷积核数目
    :param bottleneck: 是否使用bottleneck
    :param dropout_rate: dropout比率
    :param weight_decay: 权重衰减率
    :return:
    """
    concat_axis = 1 if K.image_data_format() == 'channel_first' else -1  # 确定格式

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(input_tensor)
    x = Activation('relu')(x)

    # if bottleneck:
    #     # 使用bottleneck进行降维
    #     inter_channel = filters
    #     x = Conv2D(inter_channel, (1, 1),
    #                kernel_initializer='he_normal',
    #                padding='same', use_bias=False,
    #                kernel_regularizer=l2(weight_decay))(x)
    #     x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    #     x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)

    return x


def dens_block(input_tensor, nb_filter):
    x1 = Conv_Block(input_tensor, nb_filter)
    add1 = concatenate([x1, input_tensor], axis=-1)
    x2 = Conv_Block(add1, nb_filter)
    add2 = concatenate([x1, input_tensor, x2], axis=-1)
    x3 = Conv_Block(add2, nb_filter)
    return x3


from keras.regularizers import l2


# model definition
def unet(input_shape=(128, 128, 2)):
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')
    inputs = Input(input_shape)
    # input1=Input(shape=(128,128,1))
    # input2=Input(shape=(128,128,1))
    # inputs=Concatenate(axis=-1)([input1,input2])
    # inputs = Input(shape
    # x  = Conv2D(32, 1, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    x = Conv2D(32, 7, kernel_initializer='he_normal', padding='same', strides=1, use_bias=False,
               kernel_regularizer=l2(1e-4))(inputs)
    # down first
    down1 = dens_block(x, nb_filter=64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(down1)  # 256
    # down second
    down2 = dens_block(pool1, nb_filter=64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(down2)  # 128
    # down third
    down3 = dens_block(pool2, nb_filter=128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(down3)  # 64
    # down four
    down4 = dens_block(pool3, nb_filter=256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(down4)  # 32
    # center
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # up first
    up6 = UpSampling2D(size=(2, 2))(drop5)
    # up6 = UpSampling2D(size=(2, 2))(drop5)
    add6 = concatenate([down4, up6], axis=3)
    up6 = dens_block(add6, nb_filter=256)
    # up second
    up7 = UpSampling2D(size=(2, 2))(up6)
    # up7 = UpSampling2D(size=(2, 2))(conv6)
    add7 = concatenate([down3, up7], axis=3)
    up7 = dens_block(add7, nb_filter=128)
    # up third
    up8 = UpSampling2D(size=(2, 2))(up7)
    # up8 = UpSampling2D(size=(2, 2))(conv7)
    add8 = concatenate([down2, up8], axis=-1)
    up8 = dens_block(add8, nb_filter=64)
    # up four
    up9 = UpSampling2D(size=(2, 2))(up8)
    add9 = concatenate([down1, up9], axis=-1)
    up9 = dens_block(add9, nb_filter=64)
    # output
    conv10 = Conv2D(32, 7, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    conv10 = Conv2D(2, 1, activation='softmax')(conv10)
    # model = Model(inputs=[input1,input2], outputs=conv10)
    model = Model(inputs=inputs, outputs=conv10)
    print(model.summary())
    return model

import random

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=10,
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
    model_checkpoint = ModelCheckpoint('/code/save_model/'+model_savename, monitor='loss', verbose=1,
                                       save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    Board = tf.keras.callbacks.TensorBoard(log_dir="/output/logs")

    history = model.fit(train_dataset.repeat(),
                                  steps_per_epoch=105,
                                  epochs=200,
                                  validation_data=validation_dataset,
                                #   validation_steps=50,
                                  callbacks=[model_checkpoint,
                                             early_stop,Board
                                             # Board
                                             ])
    plot_history(history, './loss_picture/', '_postCrV')
    return model

        
BATCH_SIZE=128

if __name__ == '__main__':
    save_path = '/data/caizhizheng/2D/validataset2/'
    label_path = save_path + 'label/'
    data_path = save_path + 'data/'
    output_path = save_path + 'predict/'
    label = os.listdir(label_path)
    number = []
    for f in label:
         number.append(f.split('.')[0])
    number = np.array(number, dtype=int)

    label_list=getimg(dir=label_path, oder=number, j=0)
    label_dataset = tf.data.Dataset.from_tensor_slices(label_list)
    label_dataset = label_dataset.map(onehot)
    
    img1_list=getimg(dir=data_path, oder=number, j=0)
    img2_list=getimg(dir=data_path, oder=number, j=900)
    img_list = np.stack((img1_list,img2_list),axis = -1)
    image_dataset = tf.data.Dataset.from_tensor_slices(img_list)

    train_dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    train_dataset = train_dataset.cache().shuffle(train_dataset.cardinality())

    validation_dataset = train_dataset.take(50).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    train_dataset = train_dataset.skip(50).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = unet(input_shape=(128, 128, 2))
    model_savename = 'validataset2.keras'
    train = train(model, model_savename, train_dataset, validation_dataset)
    
    image_dataset = image_dataset.batch(BATCH_SIZE)
    mask = model.predict(image_dataset)
    mask = tf.argmax(mask, axis=-1)
    mask = tf.keras.backend.eval(mask)
    mask = (mask * 255).astype(np.uint8)
    os.makedirs(output_path, exist_ok=True)
    for j in range(0, len(number)):
        cv2.imwrite(output_path+ '%d.tif' %(number[j]), mask[j])

    # name = 'error2' + '_postCrV' + '.csv'
    name = 'error' + '.csv'
    name = os.path.join(save_path, name)
    evalu.main(output_dir=output_path, label_dir=label_path, name=name, oder=number)

    # output_dir = '/data/caizhizheng/2D/validataset/predict2/'
    # output_name = '/data/caizhizheng/2D/validataset/' + 'vali-filterd2.zip'
    output_name = output_path + 'predict.zip'
    make_zip(source_dir=output_path, output_name=output_name)
    print("zip OK")