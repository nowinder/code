# coding:utf-8

import os
import shutil
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

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
import cv2
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Activation, Dropout, AveragePooling2D, concatenate, \
    GlobalAveragePooling2D, MaxPooling2D, Dense, Input
from keras.regularizers import l2
import keras.backend as K
import zipfile

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
# Read datasets
def train_generator(batch_size=32):
    data_gen_args = dict(featurewise_center=True,
                         # rotation_range=90.,
                         # width_shift_range=0.1,
                         # height_shift_range=0.1,
                         fill_mode="constant",
                         cval=255,
                         # horizontal_flip=True,
                         # vertical_flip=True,
                         zoom_range=0.2
                         )
    image1_datagen = ImageDataGenerator(**data_gen_args)
    image2_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    seed = 1
    image_generator = image1_datagen.flow_from_directory(
        '/data/caizhizheng/2D/v2/data1',
        # The absolute path of the data set
        class_mode=None,
        batch_size=batch_size,
        color_mode='grayscale',
        target_size=(128, 128),
        # save_to_dir='./data/gen/images',
        shuffle=False,
        seed=seed)
    image_generator2 = image2_datagen.flow_from_directory(
        '/data/caizhizheng/2D/v2/data2',
        # The absolute path of the data set
        class_mode=None,
        batch_size=batch_size,
        color_mode='grayscale',
        target_size=(128, 128),
        shuffle=False,
        # save_to_dir='./data/gen/images',
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        '/data/caizhizheng/2D/v2/label',
        # The absolute path of the data set
        class_mode=None,
        color_mode='grayscale',
        # color_mode='rgb',
        target_size=(128, 128),
        batch_size=batch_size,
        shuffle=False,
        # save_to_dir='./data/gen/masks',
        seed=seed)
    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, image_generator2, mask_generator)

    for (imgs1, imgs2, masks) in train_generator:
        imgs1 = imgs1 / 255.0
        imgs2 = imgs2 / 255.0
        # print(masks.shape)
        # masks = cv2.cvtColor(masks,cv2.COLOR_RGB2GRAY)
        masks = masks / 255.0
        yield (imgs1, imgs2), masks


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
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv10)
    model = Model(inputs=[input1,input2], outputs=conv10)
    print(model.summary())
    return model


# define Huber loss
def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred, delta=0.01)


def simm_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true)) + 0.01 * K.mean(K.abs(y_pred))
    # return tf.abs(tf.norm(y_pred - y_true))/tf.norm(y_true)

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


def train(model):
    # train
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=simm_loss, metrics=['accuracy'])
    reduce_lr = LearningRateScheduler(scheduler)
    model_checkpoint = ModelCheckpoint('/code/save_model/dens_2.5D.h5', monitor='loss', verbose=1,
                                       save_best_only=True)
    history = model.fit(train_generator(batch_size=256),
                                  steps_per_epoch=100,
                                  epochs=40,
                                  validation_data=train_generator(batch_size=128),
                                  validation_steps=50,
                                  callbacks=[model_checkpoint,
                                             reduce_lr
                                             ])
    plot_history(history, '.results/', 'Unet')
    return model

def test(model):
    # test
    input_dir = '/data/caizhizheng/2D/v2/predicted_input_data'
    for i in range(1,101):
        # x = cv2.imread('/data/caizhizheng/data/test/%d.tif' % (i))  # #The absolute path of the testsets
        # x = add_gaussian_nois(x)
        # x = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
        filename1 = f'{i}.tif'
        img_path1 = os.path.join(input_dir,filename1)
        x1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        x1 = x1 / 255.0
        x1 = np.array([x1])
        filename2 = f'{i+100}.tif'
        img_path2 = os.path.join(input_dir,filename2)
        x2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
        x2 = x2 / 255.0
        x2 = np.array([x2])
        # xt = np.stack([x1, x2], axis=-1)
        mask = model.predict([x1,x2], batch_size=None, verbose=0, steps=None)
        mask = mask[0]
        mask = mask * 255
        cv2.imwrite('/data/caizhizheng/2D/v2/result/%d.tif' % (i), mask)
        
def make_zip(source_dir, output_name):
    zipf = zipfile.ZipFile(output_name, 'w')
    prelen = len(os.path.dirname(source_dir))
    for parent, _, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[prelen:].strip(os.path.sep)     #相对路径
            zipf.write(pathfile, arcname)
        zipf.close()
        
if __name__ == '__main__':
    # is_train = False # you can change this to False if you want to test only
    model = unet(input_shape=(128, 128, 2))
    # if is_train:
    model = train(model) # train the model and save the best weights
    # else:
    # model.load_weights('/code/save_model/dens_2D.h5') # load the saved weights
    test(model) # test the model and save the results
    source_dir = "/data/caizhizheng/2D/v2/result"
    output_name = "/data/caizhizheng/2D/v2/result.zip"
    make_zip(source_dir=source_dir, output_name=output_name)
    print("zip OK")

        

# if __name__ == '__main__':
#     is_train = True
#     # train
    # if is_train:
    #     model = unet(input_shape=(128, 128, 1))
    #     # model.load_weights('./model/dens_dens_block_net_4.h5')
    #     model.compile(optimizer=Adam(learning_rate=0.0001), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
    #     reduce_lr = LearningRateScheduler(scheduler)
    #     model_checkpoint = ModelCheckpoint('/code/save_model/dens_2D.h5', monitor='loss', verbose=1,
    #                                        save_best_only=True)
    #     history = model.fit(train_generator(batch_size=128),
    #                                   steps_per_epoch=200,
    #                                   epochs=50,
    #                                   validation_data=train_generator(batch_size=1),
    #                                   validation_steps=50,
    #                                   callbacks=[model_checkpoint,
    #                                              reduce_lr
    #                                              ])
    #     plot_history(history, '.results/', 'Unet')
    # # test
    # else:
    #     model = unet(input_shape=(128, 128, 1))
    #     model.load_weights('/code/save_model/dens_2D.h5')
    #     # PA_all = 0
    #     # MPA_all = 0
    #     # MIOU_all = 0
    #     # ssim = 0
    #     # psnr_num = 0
    #     input_dir = '/data/caizhizheng/2D/predicted_input_data'
    #     for i in range(1,101):
    #         # x = cv2.imread('/data/caizhizheng/data/test/%d.tif' % (i))  # #The absolute path of the testsets
    #         # x = add_gaussian_nois(x)
    #         # x = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
    #         filename = f'{i}.tif'
    #         img_path = os.path.join(input_dir,filename)
    #         x = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #         x = x / 255.0
    #         x = np.array([x])
    #         mask = model.predict(x, batch_size=None, verbose=0, steps=None)
    #         mask = mask[0]
    #         mask = mask * 255
    #         cv2.imwrite('/data/caizhizheng/2D/result/%d.tif' % (i), mask)
            # Evaluation function
            # img2 = cv2.imread('/code/caizhizheng/2D/result/%d.tif' % (i))
            # img = cv2.imread('/data/caizhizheng/2D/predicted_label/%d.tif' % (i))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            # use cv2.absdiff and np.linalg.norm to get relative residual
#            diff = cv2.absdiff(img,img2)
#            rel_res = np.linalg.norm(diff) / np.linalg.norm(img)
#            rel_res = rel_res[0]
#            rel_res = rel_res * 255
#            cv2.imwrite('/code/loss_picture/rel_res/rer_%d.png'% (i), rel_res)
        #     ssim += compute_ssim(np.array(img), np.array(img2))
        #     psnr_num += psnr(img,img2)
        # ssim = ssim/100
        # psnr_num = psnr_num / 100
        # print(f'网络预测与label之间ssim, psnr平均值分别为: {ssim:}, {psnr_num:}')
