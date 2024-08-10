import os
# import shutil
# import skimage.io as io
# import skimage.transform as trans
# from keras.optimizer_v2.adam import Adam
# from keras.preprocessing.image import ImageDataGenerator
import numpy as np
# from keras.models import *
# from keras.layers import *
# from keras.optimizers import *
import keras.optimizers as optimizer
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras import backend as keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import imageio.v2 as imageio
# from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Activation, Dropout, AveragePooling2D, concatenate, \
    GlobalAveragePooling2D, MaxPooling2D, Dense, Input, UpSampling2D
from keras.regularizers import l2
import keras.backend as K
import zipfile
import random
import evalu
from nd_mlp_mixer import MLPMM
# import time
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.config.experimental.set_memory_growth(gpus[1], True)
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
# Read datasets
def getimg(dir,num,j):
    input_dir = dir
    img_list = []
    for i in range(1+j,num+1+j):
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
    # concat_axis = 1 if K.image_data_format() == 'channel_first' else -1  # 确定格式

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(input_tensor)
    x = Activation('relu')(input_tensor)

    # if bottleneck:
    #     # 使用bottleneck进行降维
    #     inter_channel = filters
    #     x = Conv2D(inter_channel, (1, 1),
    #                kernel_initializer='he_normal',
    #                padding='same', use_bias=False,
    #                kernel_regularizer=l2(weight_decay))(x)
    #     x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    #     x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=True)(x)
    # x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', use_bias=True)(x)

    return x

# 定义一个函数来添加随机通道
def add_random_channel(inputs):
    batch_size, height, width = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
    random_channel = tf.random.uniform(shape=(batch_size, height, width, 1), minval=0, maxval=1)
    return tf.concat([inputs, random_channel], axis=-1)

def dens_block(input_tensor, nb_filter):
    x1 = Conv_Block(input_tensor, nb_filter)
    add1 = concatenate([x1, input_tensor], axis=-1)
    x2 = Conv_Block(add1, nb_filter)
    add2 = concatenate([x1, input_tensor, x2], axis=-1)
    x3 = Conv_Block(add2, nb_filter)
    return x3

def MMM(input_shape=(128,128,2)):
    inputs = Input(input_shape)
    arc = add_random_channel(inputs)
    x = MLPMM(num_blocks=10, patch_size=16, tokens_mlp_dim=128, channels_mlp_dim=1024)(arc)
    model = Model(inputs=inputs,outputs=x)
    print(model.summary())
    return model

def unet(input_shape=(128, 128, 2)):
    # tf.keras.mixed_precision.set_global_policy('mixed_float16')
    inputs = Input(input_shape)
    arc = add_random_channel(inputs)
    # input1=Input(shape=(128,128,1))
    # input2=Input(shape=(128,128,1))
    # inputs=Concatenate(axis=-1)([input1,input2])
    # inputs = Input(shape
    # x  = Conv2D(32, 1, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    x = Conv2D(32, 7, kernel_initializer='he_normal', padding='same', strides=1, use_bias=True,
               kernel_regularizer=l2(0))(arc)
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
    conv10 = Conv2D(2, 1, activation='softmax',use_bias=False)(conv10)
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



lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=10,
  decay_rate=1,
  staircase=False)

# Define the learning rate attenuation value
def scheduler(epoch):
    # if epoch % 10 == 0 and epoch != 0:
    #     lr = K.get_value(model.optimizer.lr)
    #     K.set_value(model.optimizer.lr, lr * 0.1)
    #     print("lr change to {}".format(lr * 0.1))
    # return K.get_value(model.optimizer.lr)
    if epoch >=50:
        new_lr = lr_schedule(epoch-50)
        K.set_value(model.optimizer.lr, new_lr)
    return K.get_value(model.optimizer.lr)



        
def make_zip(source_dir, output_name):
    zipf = zipfile.ZipFile(output_name, 'w')
    prelen = len(os.path.dirname(source_dir))
    for parent, _, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[prelen:].strip(os.path.sep)     #相对路径
            zipf.write(pathfile, arcname)
        zipf.close()

def train(model, model_savename, train_dataset, validation_dataset,reduce_lr):
    # model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    # reduce_lr = LearningRateScheduler(scheduler)
    # reduce_lr = LearningRateScheduler(lschedule)
    # model.load_weights('/code/save_model/exped/'+model_savename)
    model_checkpoint = ModelCheckpoint('G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/save_model/exped/'+model_savename,save_weights_only=True, monitor='loss', verbose=1,
                                       save_best_only=True, initial_value_threshold=0.002)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
    Board = tf.keras.callbacks.TensorBoard(log_dir="G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/output/logs_mlpmixer",histogram_freq=100)
    
    history = model.fit(train_dataset,
                                  steps_per_epoch=100,
                                  epochs=1000,  
                                  validation_data=validation_dataset,
                                #   validation_steps=50,
                                  callbacks=[model_checkpoint,
                                             early_stop,Board,reduce_lr
                                             # Board
                                             ])
    plot_history(history, 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/code/loss_picture/',prefix='_exped')
    return model
    
def test(model, test_dataset, output_path, length):
    mask = model.predict(test_dataset)
    mask = tf.argmax(mask, axis=-1)
    mask = tf.keras.backend.eval(mask)
    mask = (mask * 255).astype(np.uint8)
    os.makedirs(output_path, exist_ok=True)
    for j in range(0, length):
        cv2.imwrite(output_path+ '%d.tif' %(j+1), mask[j])
    print('model predict OK')
        
BATCH_SIZE=4
plot_path = '/code/loss_picture/extended/'

if __name__ == '__main__':
    NUM_train = 3900
    NUM_tv = 200
    output_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/exped_output/'
    label_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/label/label/'
    img_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/data1/data/'
    img1_list = getimg(img_path,num=NUM_train, j=0)
    img_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/data2/data/'
    img2_list = getimg(img_path,num=NUM_train, j=900)
    label_list = getimg(label_path,num=NUM_train, j=0)
    img_list = np.stack((img1_list,img2_list),axis = -1)
    
    img_path2 = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/test/predicted_input_data/'
    label_path2 = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/test/predicted_label/'
    # output_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/test/outputv2/'
    img1_list = getimg(img_path2,num=100, j =0)
    img2_list = getimg(img_path2,num=100, j =100)
    test_img_list = np.stack((img1_list,img2_list),axis = -1)
    test_label_list = getimg(label_path2, num=100, j=0)
    
    img_list = np.vstack((img_list, test_img_list))
    label_list = np.vstack((label_list, test_label_list))
    
    image_dataset = tf.data.Dataset.from_tensor_slices(img_list)
    label_dataset = tf.data.Dataset.from_tensor_slices(label_list)
    label_dataset = label_dataset.map(onehot)
    # 使用事先保存的数据集会造成内存泄漏？？
    # image_dataset = tf.data.experimental.load("dataset/image_exped")
    # label_dataset = tf.data.experimental.load("dataset/label_exped")
    # x = next(iter(image_dataset.batch(image_dataset.cardinality().numpy())))
    # y = next(iter(label_dataset.batch(label_dataset.cardinality().numpy())))
    origin_dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    # origin_dataset = origin_dataset.shuffle(origin_dataset.cardinality(), seed=73)

    # validation_dataset = origin_dataset.take(100).batch(BATCH_SIZE)
    # validation_dataset = validation_dataset.cache().prefetch(tf.data.AUTOTUNE)
    train_dataset = origin_dataset
    
    se =random.randint(1,int(10e2))
    print('seed=%d' % se)
    train_dataset = train_dataset.cache().shuffle(train_dataset.cardinality(),seed=se, reshuffle_each_iteration=True).repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    test_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/test/test2/data/'
    test_label_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/test/test2/label/'
    test_output_path = 'G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/data_2D/v2/test/outputv2/'
    img1_list = getimg(test_path,num=NUM_tv, j =0)
    img2_list = getimg(test_path,num=NUM_tv, j =200)
    img_list = np.stack((img1_list,img2_list),axis = -1)
    label_list = getimg(test_label_path,num=NUM_tv,j=0)
    tv_imgset = tf.data.Dataset.from_tensor_slices(img_list)
    tv_labelset = tf.data.Dataset.from_tensor_slices(label_list)
    tv_labelset = tv_labelset.map(onehot)
    tv_set = tf.data.Dataset.zip((tv_imgset, tv_labelset))
    validation_dataset = tv_set.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

    
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    # model = unet(input_shape=(128, 128, 2))
    model = MMM()
    # model = NdAutoencoder(in_shape=(128,128,2),repr_shape=(16,16),num_mix_layers=10,hidden_size=256)
    model.compile(optimizer=optimizer.Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    reduce_lr = LearningRateScheduler(scheduler)
        # model.load_weights('/code/save_model/exped/cross_ini_277.keras')
    model_savename = '8_10_3900dunetbase.keras'
    train = train(model, model_savename, train_dataset, validation_dataset,reduce_lr)
    
    image_dataset = image_dataset.batch(BATCH_SIZE)
    test(model, image_dataset, output_path, NUM_train+100)
    # name = 'error2' + '_postCrV' + '.csv'
    error_path = '../errors/'
    name = 'error_expedv2' + '.csv'
    name = os.path.join(error_path, name)
    evalu.main(output_dir=output_path, label_dir=label_path, name=name, oder=np.array(0))
    
#     测试集

    # test_dataset = tf.data.Dataset.from_tensor_slices(img_list)
    test_dataset = tv_imgset.batch(1)
    try:
        model.load_weights('G:/Zheng_caizhi/Pycharmprojects/IC_inverseimage/save_model/exped/'+model_savename)
    except: pass
    test(model, test_dataset, test_output_path,NUM_tv)
    name = 'error_exped_testv2.csv'
    name = os.path.join(error_path, name)
    evalu.main(output_dir=test_output_path, label_dir=test_label_path, name=name, oder=np.array(0))
    print('seed=%d' % se)