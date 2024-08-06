# coding:utf-8
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
# def train_generator(label_dataset, batch_size=32):
#     image1_datagen = ImageDataGenerator()
#     image2_datagen = ImageDataGenerator()
#     mask_datagen = ImageDataGenerator()
#     seed=1
#     def image_generator():
#         image_generator1 = image1_datagen.flow_from_directory(
#             '/data/caizhizheng/2D/v2/data1',
#             # The absolute path of the data set
#             class_mode=None,
#             batch_size=batch_size,
#             color_mode='grayscale',
#             target_size=(128, 128),
#             # save_to_dir='./data/gen/images',
#             shuffle=False,
#             seed=seed)
#         image_generator2 = image2_datagen.flow_from_directory(
#             '/data/caizhizheng/2D/v2/data2',
#             # The absolute path of the data set
#             class_mode=None,
#             batch_size=batch_size,
#             color_mode='grayscale',
#             target_size=(128, 128),
#             shuffle=False,
#             # save_to_dir='./data/gen/images',
#             seed=seed)
#         for imgs1, imgs2 in zip(image_generator1, image_generator2):
#             imgs1 = imgs1 / 255.0
#             imgs2 = imgs2 / 255.0
#             yield (imgs1, imgs2)
            
#     def mask_generator():
#         for masks in mask_datagen.flow(label_dataset, batch_size=batch_size):
#             yield masks
    
#     train_dataset = tf.data.Dataset.from_generator(
#         lambda: zip(image_generator(), mask_generator()),
#         output_types=((tf.float32, tf.float32), tf.float32),
#         output_shapes=(((batch_size, 128, 128, 1), (batch_size, 128, 128, 1)), (batch_size, 128, 128, 2))
#     )
    
#     return train_dataset


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
    plt.legend(['acc', 'val_acc'], loc='upper right')
	# plt.show()
    plt.savefig('/code/loss_picture/unet_val_ace.png')
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    # plt.show()
    plt.savefig('/code/loss_picture/denseunet_loss.png')
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
    # inputs = Input(input_shape)
    input1=Input(shape=(128,128,1))
    input2=Input(shape=(128,128,1))
    inputs=Concatenate(axis=-1)([input1,input2])
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
    model = Model(inputs=[input1,input2], outputs=conv10)
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


# ssim psnr
from ssim import compute_ssim
import math


def psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


import random


def add_gaussian_nois(image_in, mean=0, var=0.01):
    """
    给图片添加高斯噪声
    """
    img = image_in.astype(np.int16)
    mu = 0
    sigma = 40
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                img[i, j, k] = img[i, j, k] + random.gauss(mu=mu, sigma=sigma)
    img[img > 255] = 255
    img[img < 0] = 0
    img_out = img.astype(np.uint8)

    # cv2.imshow("noise_image",img_out)
    # cv2.waitKey(0)
    return img_out

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=10,
  decay_rate=1,
  staircase=False)
def train(model):
    # train
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')
    # no shutil and shutil module
    # logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
    # shutil.rmtree(logdir, ignore_errors=True)
    # use tf.io.gfile
    # logdir = tf.io.gfile.mkdir('/tensorboard_logs')
    # tf.io.gfile.rmtree (logdir)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    # reduce_lr = LearningRateScheduler(scheduler)
    # reduce_lr = LearningRateScheduler(lschedule)
    model_checkpoint = ModelCheckpoint('/code/save_model/'+model_savename, monitor='loss', verbose=1,
                                       save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    Board = tf.keras.callbacks.TensorBoard(log_dir="/output/logs")

    history = model.fit(train_dataset.repeat(),
                                  steps_per_epoch=100,
                                  epochs=400,
                                  validation_data=validation_dataset,
                                #   validation_steps=50,
                                  callbacks=[model_checkpoint,
                                             early_stop,Board
                                             ])
    plot_history(history, '.results/', 'Unet')
    return model

def test(model):
    # test
    input_dir1 = input_dir2 = '/data/caizhizheng/2D/v2/predicted_input_data'
    # output_dir = '/data/caizhizheng/2D/v2/result/'
    n = len(os.listdir(input_dir1))
    for i in range(1,n//2+1):
        # x = cv2.imread('/data/caizhizheng/data/test/%d.tif' % (i))  # #The absolute path of the testsets
        # x = add_gaussian_nois(x)
        # x = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
        filename1 = f'{i}.tif'
        img_path1 = os.path.join(input_dir1,filename1)
        x1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        x1 = x1 / 255.0
        x1 = np.array([x1])
        filename2 = f'{i+n//2}.tif'
        img_path2 = os.path.join(input_dir2,filename2)
        x2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
        x2 = x2 / 255.0
        x2 = np.array([x2])
        # xt = np.stack([x1, x2], axis=-1)
        mask_tensor = model.predict([x1,x2], batch_size=None, verbose=0, steps=None)
        mask = mask_tensor[0]
        mask = tf.argmax(mask, axis=-1)
        mask = tf.keras.backend.eval(mask)
        mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(output_dir+'%d.tif' % (i), mask)
        
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
if __name__ == '__main__':
    # is_train = False # you can change this to False if you want to test only
    model = unet(input_shape=(128, 128, 2))
    # model = tf.keras.models.load_model('/code/save_model/dens_2.5Dv3.keras')
    # if is_train:
    label_list=getimg(dir='/data/caizhizheng/2D/v2/label/label',j=0)
    label_dataset = tf.data.Dataset.from_tensor_slices(label_list)
    label_dataset = label_dataset.map(onehot)
    img1_list=getimg(dir='/data/caizhizheng/2D/v2/data1/data',j=0)
    img1_dataset = tf.data.Dataset.from_tensor_slices(img1_list)
    # img1_dataset = img1_dataset.map(add_channel)
    img2_list=getimg(dir='/data/caizhizheng/2D/v2/data2/data',j=900)
    img2_dataset = tf.data.Dataset.from_tensor_slices(img2_list)
    # img2_dataset = img2_dataset.map(add_channel)
    image_dataset = tf.data.Dataset.zip((img1_dataset, img2_dataset))
    train_dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    #4th modifation
    train_dataset = train_dataset.shuffle(buffer_size=len(train_dataset))
    validation_dataset = train_dataset.take(32).cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    train_dataset = train_dataset.skip(32).cache()
    
    # train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BATCH_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    # train_dataset = train_dataset.batch(32)

    time1 = time.strftime('%m%d%H')
    model_savename = 'dens_2.5Dv3' + time1 +'.keras'
    result_name = 'result' + time1 + '.zip'
    output_dir = "/data/caizhizheng/2D/v2/result"
    output_name = "/data/caizhizheng/2D/v2/"+result_name
    
    model = train(model) # train the model and save the best weights
    # else:
    # model.load_weights('/code/save_model/dens_2.5Dv3.keras') # load the saved weights
    test(model) # test the model and save the results
    
    make_zip(source_dir=output_dir, output_name=output_name)
    print("zip OK")
    evalu.main(output_dir=output_dir, label_dir='/data/caizhizheng/2D/v2/predicted_label/')

