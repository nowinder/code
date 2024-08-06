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

from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from keras.utils.np_utils import to_categorical
import zipfile
import evalu
import time
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
# Read datasets
def getimg(dir,j):
    input_dir = dir
    img_list = []
    for i in range(1+j,901+j):
        filename = f'{i}.tif'
        img_path = os.path.join(input_dir,filename)
        img = imageio.imread(img_path)
        img = img.astype(np.float32)
        img = img / 255.0
        img_list.append(img)
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
    plt.legend(['acc', 'val_acc'], loc='upper right')
	# plt.show()
    plt.savefig(result_dir + 'unet_ace' + prefix + '.png')
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale("log")
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    # plt.show()
    plt.savefig(result_dir + 'unet_loss'+ prefix +'.png')
    plt.close()

    # x = history.history['loss']
    # np.savetxt('D:/pycharm/up_down_code/loss_picture/unet_train_loss.txt', x, fmt='%f')
    # y = history.history['val_loss']
    # np.savetxt('D:/pycharm/up_down_code/loss_picture/unet_val_loss.txt', y, fmt='%f')hb


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


# define Huber loss
def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred, delta=0.01)


def simm_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true)) + 0.01 * K.mean(K.abs(y_pred))
    # return tf.abs(tf.norm(y_pred - y_true))/tf.norm(y_true)
    
# smooth = 1. # 用于防止分母为0.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true) # 将 y_true 拉伸为一维.
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f))

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

# Define the learning rate attenuation value
def scheduler(epoch):
    if epoch % 10 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr change to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)

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
        
BATCH_SIZE=128
plot_path = '/code/loss_picture/CrV/'
fold_no = 1

if __name__ == '__main__':
    # is_train = False # you can change this to False if you want to test only
    model = unet(input_shape=(128, 128, 2))
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    # model_checkpoint = ModelCheckpoint('/code/save_model/'+model_savename, monitor='loss', verbose=1,
    #                                    save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)
    Board = tf.keras.callbacks.TensorBoard(log_dir="/output/logs")

    image_dataset = tf.data.experimental.load("dataset/data1")
    label_dataset = tf.data.experimental.load("dataset/label")
    x = next(iter(image_dataset.batch(image_dataset.cardinality().numpy())))
    y = next(iter(label_dataset.batch(label_dataset.cardinality().numpy())))
    
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(x,y):
        x_train, x_test = tf.gather(x, train_index), tf.gather(x, test_index)
        y_train, y_test = tf.gather(y, train_index), tf.gather(y, test_index)
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        vali_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        train_dataset = train_dataset.cache().shuffle(train_dataset.cardinality()).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        vali_dataset = vali_dataset.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        model_savename = 'CrossVali' + str(fold_no) +'.keras'
        model_checkpoint = ModelCheckpoint('/code/save_model/CrossVali2/'+model_savename, monitor='loss', verbose=1,
                                           save_best_only=True)
        history = model.fit(train_dataset.repeat(), 
                                      steps_per_epoch=100,
                                      epochs=400,
                                      validation_data=vali_dataset,
                                    #   validation_steps=50,
                                      callbacks=[model_checkpoint,
                                                 early_stop,Board
                                                 ])
            
        
        plot_history(history, plot_path, str(fold_no))
        fold_no = fold_no +1
        print(f"Fold {fold_no} train complete")