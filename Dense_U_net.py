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
import scipy.io

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

# Read datasets
#def train_generator(batch_size):
#    data_gen_args = dict(featurewise_center=True,
                         # rotation_range=90.,
                         # width_shift_range=0.1,
                         # height_shift_range=0.1,
#                         fill_mode="constant",
#                         cval=0,
                         # horizontal_flip=True,
                         # vertical_flip=True,
                         # zoom_range=0.2
#                         )
   
#    image_datagen = ImageDataGenerator(**data_gen_args)
#    mask_datagen = ImageDataGenerator(**data_gen_args)

#    seed = 1 
#    imgs = []
#    masks = []
#    for ch in range(1, 21):  
#        image_gen = image_datagen.flow_from_directory(
#                f'/data/caizhizheng/data/data/',
#                class_mode=None,
#                color_mode='grayscale',
#                target_size=(32, 32),
#                batch_size=900, 
#                shuffle=False,    
#                seed=seed)
#        imgs.append(image_gen)

#        mask_gen = mask_datagen.flow_from_directory(
#                f'/data/caizhizheng/data/label/',
#                class_mode=None,
#                color_mode='grayscale',  
#                target_size=(32, 32),
#                batch_size=900,
#                shuffle=False,
#                seed=seed) 
#        masks.append(mask_gen)

    # 将20个通道的imgs和masks在第三维度上concat
#    imgs = np.concatenate(imgs, axis=-1)   
#    masks = np.concatenate(masks, axis=-1) 

#    yield imgs, masks  

#    image_datagen = ImageDataGenerator(**data_gen_args)
#    mask_datagen = ImageDataGenerator(**data_gen_args)
#    seed = 1
#    image_generator = image_datagen.flow_from_directory(
#        '/data/caizhizheng/data/data',
        # The absolute path of the data set
#        class_mode=None,
#        batch_size=batch_size,
#        color_mode='grayscale',
#        target_size=(20, 20),
        # save_to_dir='./data/gen/images',
#        seed=seed)
#    mask_generator = mask_datagen.flow_from_directory(
#        '/data/caizhizheng/data/label',
        # The absolute path of the data set
#        class_mode=None,
#        color_mode='grayscale',
        # color_mode='rgb',
#        target_size=(20, 20),
#        batch_size=batch_size,
        # save_to_dir='./data/gen/masks',
#        seed=seed)
    # combine generators into one which yields image and masks
#    train_generator = zip(image_generator, mask_generator)

#    for (imgs, masks) in train_generator:
#        imgs = imgs / 255.0
        # print(masks.shape)
        # masks = cv2.cvtColor(masks,cv2.COLOR_RGB2GRAY)
#        masks = masks / 255.0
#        yield imgs, masks
	# Create an empty list to store the matrices

matrices = []
    # Loop through the mat files and load them into the list
for i in range(1, 901):
    # Load the mat file into a dictionary
    mat = scipy.io.loadmat(f'/data/caizhizheng/data/ndata/data/{i}.mat')
     # Extract the matrix value from the dictionary
    matrix = mat['img_merge']
    matrix = matrix.astype(np.float32)
    # Append the matrix to the list
    matrices.append(matrix)

    # Stack the matrices into a numpy array
matrices = np.stack(matrices)

    # Create a tf.data.Dataset object from the numpy array
dataset_img = tf.data.Dataset.from_tensor_slices(matrices)
matrices = []
for i in range(1, 901):
    mat = scipy.io.loadmat(f'/data/caizhizheng/data/nlabel/label/{i}.mat')
    matrix = mat['img_merge']
    matrix = matrix.astype(np.float32)
    matrices.append(matrix)

matrices = np.stack(matrices)
dataset_label = tf.data.Dataset.from_tensor_slices(matrices)
datasets = tf.data.Dataset.zip((dataset_img,dataset_label))
#datasets = tf.data.Dataset.from_tensor_slices((dataset_img, dataset_label))
def normalize_datasets(datasets, batch_size):
    imgs_batch = []
    masks_batch = []
    for (imgs, masks) in datasets:
        imgs = imgs / 255.0
        masks = masks / 255.0
        imgs_batch.append(imgs)
        masks_batch.append(masks)
        if len(imgs_batch) == batch_size: # 当达到批次大小时
            yield np.stack(imgs_batch), np.stack(masks_batch) # 返回一个批次的数据
            imgs_batch = [] # 清空列表
            masks_batch = []
    if len(imgs_batch) > 0: # 如果还有剩余的数据
        yield np.stack(imgs_batch), np.stack(masks_batch) # 返回最后一个批次的数据







# Draw loss curve
def plot_history(history, result_dir, prefix):
    """
    将训练与验证的accuracy与loss画出来
    """
    #plt.plot(history.history['accuracy'], marker='.')
    #plt.plot(history.history['val_accuracy'], marker='.')
    #plt.title('model accuracy')
   # plt.xlabel('epoch')
    #plt.ylabel('accuracy')
    #plt.grid()
    #plt.legend(['acc', 'val_acc'], loc='upper right')
	# plt.show()
    #plt.savefig('/code/loss_picture/unet_val_ace.png')
    #plt.close()

    plt.plot(history.history['loss'], marker='.')
    #plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss'], loc='upper right')
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
def unet(input_shape=(32, 32, 20)):
    inputs = Input(input_shape)
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
 #   up7 = Conv2D(128, (3,3), activation='relu', padding='same')(up7)  # channel调整为128
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
    conv10 = Conv2D(20, 1, activation='sigmoid')(conv10)
    model = Model(inputs=inputs, outputs=conv10)
    print(model.summary())
    return model


# define Huber loss
def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred, delta=0.01)


def simm_loss(y_true, y_pred):
    return tf.abs(tf.norm(y_pred - y_true))/tf.norm(y_true)


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


if __name__ == '__main__':
    is_train = True
    # train
    if is_train:
        #datasets = datasets.batch(32)
        
        # Split the dataset into train and validation sets
        train_dataset = datasets.take(640)
        val_dataset = datasets.skip(640)
        train_dataset = normalize_datasets(train_dataset, 1)
        val_dataset=normalize_datasets(val_dataset, 1)
        model = unet(input_shape=(32, 32, 20))
        # model.load_weights('./model/dens_dens_block_net_4.h5')
        model.compile(optimizer=Adam(learning_rate=0.001), loss=simm_loss, metrics=['accuracy'])
        reduce_lr = LearningRateScheduler(scheduler)
        model_checkpoint = ModelCheckpoint('/code/save_model/dens_dens_block_net_5.h5', monitor='loss', verbose=1,
                                           save_best_only=True)
        history = model.fit(train_dataset,
                                      epochs=20,
                                      steps_per_epoch=200,
                                      validation_data=val_dataset,
                                      callbacks=[model_checkpoint,
                                                 reduce_lr
                                                 ])
        plot_history(history, '.results/', 'Unet')

    else:
        model = unet(input_shape=(32, 32, 20))
        model.load_weights('/code/save_model/dens_dens_block_net_5.h5')
        PA_all = 0
        MPA_all = 0
        MIOU_all = 0
        ssim = 0
        psnr_num = 0
        input_dir = '/data/caizhizheng/data/test/'
        for i in range(1,101):
            # x = cv2.imread('/data/caizhizheng/data/test/%d.tif' % (i))  # #The absolute path of the testsets
            # x = add_gaussian_nois(x)
            # x = cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
            filename = f'{i}.tif'
            img_path = os.path.join(input_dir,filename)
            x = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            x = x / 255.0
            x = np.array([x])
            mask = model.predict(x, batch_size=None, verbose=0, steps=None)
            mask = mask[0]
            mask = mask * 255
            cv2.imwrite('/code/predict_data/denseunet_%d.png' % (i), mask)
            # Evaluation function
            img2 = cv2.imread('/code/predict_data/denseunet_%d.png' % (i))
            img = cv2.imread('/data/caizhizheng/data/test_label/%d.tif' % (i))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            # use cv2.absdiff and np.linalg.norm to get relative residual
#            diff = cv2.absdiff(img,img2)
#            rel_res = np.linalg.norm(diff) / np.linalg.norm(img)
#            rel_res = rel_res[0]
#            rel_res = rel_res * 255
#            cv2.imwrite('/code/loss_picture/rel_res/rer_%d.png'% (i), rel_res)
            ssim += compute_ssim(np.array(img), np.array(img2))
            psnr_num += psnr(img,img2)
        ssim = ssim/100
        psnr_num = psnr_num / 100
        print(f'网络预测与label之间ssim, psnr平均值分别为: {ssim:}, {psnr_num:}')

