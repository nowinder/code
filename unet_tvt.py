import os
# import shutil
# import skimage.io as io
# import skimage.transform as trans
from keras.optimizer_v2.adam import Adam
# from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from keras import backend as keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import imageio.v2 as imageio
# from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Activation, Dropout, AveragePooling2D, concatenate, \
    GlobalAveragePooling2D, MaxPooling2D, Dense, Input
from keras.regularizers import l2
import keras.backend as K
import zipfile
import evalu
from keras.regularizers import l2
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
# Read datasets
def getimg(dir,num,j):
    input_dir = dir
    img_list = []
    for i in range(1+j,num+j):
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
    # print(model.summary())
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
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr change to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)


lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=1000,
  decay_rate=1,
  staircase=False)

# reduce_lr = LearningRateScheduler(scheduler)

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
    # model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    reduce_lr = LearningRateScheduler(scheduler)
    # reduce_lr = LearningRateScheduler(lschedule)
    # model.load_weights('/code/save_model/exped/'+model_savename)
    model_checkpoint = ModelCheckpoint('/code/save_model/exped/'+model_savename, monitor='val_loss', verbose=1,
                                       save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)
    Board = tf.keras.callbacks.TensorBoard(log_dir="/output/logs")
    
    history = model.fit(train_dataset.repeat(),
                                  steps_per_epoch=100,
                                  epochs=3000,
                                  validation_data=validation_dataset,
                                #   validation_steps=50,
                                  callbacks=[model_checkpoint,
                                             early_stop,Board, reduce_lr
                                             ])
    plot_history(history, '/code/loss_picture/extended/',prefix='_exped')
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
        
BATCH_SIZE=128
plot_path = '/code/loss_picture/extended/'

if __name__ == '__main__':
    output_path = '/data/caizhizheng/2D/v2/vali_output/'
    label_path = '/data/caizhizheng/2D/v2/label/label/'
    img_path = '/data/caizhizheng/2D/v2/data1/data/'
    img1_list = getimg(img_path,num=2901, j=0)
    img_path = '/data/caizhizheng/2D/v2/data2/data/'
    img2_list = getimg(img_path,num=2901, j=900)
    label_list = getimg(label_path,num=2901, j=0)
    img_list = np.stack((img1_list,img2_list),axis = -1)
    
    vali_path = '/data/caizhizheng/2D/v2/test/predicted_input_data/'
    vali_label_path = '/data/caizhizheng/2D/v2/test/predicted_label/'
    # output_path = '/data/caizhizheng/2D/v2/test/outputv2/'
    img1_list = getimg(vali_path,num=101, j =0)
    img2_list = getimg(vali_path,num=101, j =100)
    vali_list = np.stack((img1_list,img2_list),axis = -1)
    vali_label_list = getimg(vali_label_path, num=101, j=0)
    
    # img_list = np.vstack((img_list, vali_list))
    # label_list = np.vstack((label_list, vali_label_list))
    
    image_dataset = tf.data.Dataset.from_tensor_slices(img_list)
    label_dataset = tf.data.Dataset.from_tensor_slices(label_list)
    label_dataset = label_dataset.map(onehot)
    train_dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
    train_dataset = train_dataset.cache().shuffle(train_dataset.cardinality()).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    vali_img_dataset = tf.data.Dataset.from_tensor_slices(vali_list)
    vali_label_dataset = tf.data.Dataset.from_tensor_slices(vali_label_list)
    vali_label_dataset = vali_label_dataset.map(onehot)
    vali_dataset = tf.data.Dataset.zip((vali_img_dataset, vali_label_dataset))
    vali_dataset = vali_dataset.cache().shuffle(vali_dataset.cardinality()).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    # 使用事先保存的数据集会造成内存泄漏？？
    # image_dataset = tf.data.experimental.load("dataset/image_exped")
    # label_dataset = tf.data.experimental.load("dataset/label_exped")
    # x = next(iter(image_dataset.batch(image_dataset.cardinality().numpy())))
    # y = next(iter(label_dataset.batch(label_dataset.cardinality().numpy())))
    
    # validation_dataset = train_dataset.take(50).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    # train_dataset = train_dataset.skip(50).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = unet(input_shape=(128, 128, 2))
        model.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        # model.load_weights('/code/save_model/exped/cross_ini_277.keras')
    model_savename = 'tvt_1900.keras'
    train = train(model, model_savename, train_dataset, vali_dataset)
    
    # image_dataset = image_dataset.batch(BATCH_SIZE)
    test(model, vali_dataset, output_path, 100)
    # name = 'error2' + '_postCrV' + '.csv'
    error_path = './errors/'
    name = 'error_vali100' + '.csv'
    name = os.path.join(error_path, name)
    evalu.main(output_dir=output_path, label_dir=vali_label_path, name=name, oder=np.array(0))
    
#     测试集
    test_path = '/data/caizhizheng/2D/v2/test/test2/data/'
    test_label_path = '/data/caizhizheng/2D/v2/test/test2/label/'
    output_path = '/data/caizhizheng/2D/v2/test/outputv2/'
    img1_list = getimg(test_path,num=201, j =0)
    img2_list = getimg(test_path,num=201, j =200)
    img_list = np.stack((img1_list,img2_list),axis = -1)
    test_dataset = tf.data.Dataset.from_tensor_slices(img_list)
    test_dataset = test_dataset.batch(1)
    model.load_weights('/code/save_model/exped/'+model_savename)
    test(model, test_dataset, output_path,200)
    name = 'error_test200.csv'
    name = os.path.join(error_path, name)
    evalu.main(output_dir=output_path, label_dir=test_label_path, name=name, oder=np.array(0))
    