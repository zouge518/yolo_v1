import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from utils import config
configg=config.Config()
import ops

def resdul(img_input, filters=64, num_row=3, num_col=3, padding='same', strides=(1, 1),bn=True,activation='lrelu'):
    # 定义残差块
    x=conv2d_bn(x=img_input, filters=filters, num_row=num_row, num_col=num_col, padding=padding, strides=strides,bn=bn,activation=activation)
    x=conv2d_bn(x=x, filters=filters, num_row=num_row, num_col=num_col, padding=padding, strides=strides,bn=bn,activation=activation)
    x=tf.add(x,img_input)
    return x
def conv2d_bn(x=None, filters=64, num_row=3, num_col=3, padding='same', strides=(2, 2), name=None, bn=True,
              activation='relu'):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = layers.Conv2D(filters=filters, kernel_size=(num_row, num_col),
                      strides=strides,
                      padding=padding,
                      use_bias=False,
                      kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                      name=conv_name)(x)
    if (bn):
        x = layers.BatchNormalization(axis=3, scale=False, name=bn_name)(x)
    if (activation == 'relu'):
        x = layers.Activation(keras.activations.relu, name=name)(x)
    elif (activation == 'linear'):
        x = layers.Activation(keras.activations.linear, name=name)(x)
    elif (activation == 'softmax'):
        x = layers.Activation(keras.activations.softmax, name=name)(x)
    elif (activation == 'sigmoid'):
        x = layers.Activation(keras.activations.sigmoid, name=name)(x)
    elif(activation == 'lrelu'):
         x=tf.nn.leaky_relu(x)
    else:
        return x
    return x


def head_network(inputs=None):
    x = conv2d_bn(x=inputs, filters=64, num_row=3, num_col=3, padding='same', strides=(1, 1),activation='relu')
    x = conv2d_bn(x=x, filters=64, num_row=3, num_col=3, padding='same', strides=(2, 2),activation='relu')
    # 缩小第一次
    x = conv2d_bn(x=x, filters=128, num_row=3, num_col=3, padding='same', strides=(1, 1),activation='relu')
    x = conv2d_bn(x, filters=128, num_row=3, num_col=3, padding='same', strides=(2, 2),activation='relu')
    # 缩小第二次
    x = conv2d_bn(x, filters=256, num_row=3, num_col=3, padding='same', strides=(1, 1),activation='relu')
    x = conv2d_bn(x=x, filters=256, num_row=3, num_col=3, padding='same', strides=(2, 2),activation='relu')
    # 缩小第三次
    x = resdul(x, filters=256, num_row=3, num_col=3, padding='same', strides=(1, 1),activation='relu')
    x = conv2d_bn(x=x, filters=256, num_row=3, num_col=3, padding='same', strides=(2, 2),activation='relu')
    # 缩小第四次
    x = conv2d_bn(x, filters=512, num_row=3, num_col=3, padding='same', strides=(1, 1))
    x = conv2d_bn(x=x, filters=512, num_row=3, num_col=3, padding='same', strides=(2, 2))
    # 缩小第五次
    x = conv2d_bn(x, filters=1024, num_row=3, num_col=3, padding='same', strides=(1, 1))
    x = conv2d_bn(x=x, filters=1024, num_row=3, num_col=3, padding='same', strides=(2, 2))
    # 缩小第6次
    return x


def boungding_and_classlayer(features,num_class=19,boudings=2,):
    x = conv2d_bn(x=features, filters=1024, num_row=3, num_col=3, padding='same', strides=(1, 1), bn=False,activation='relu')
    x = conv2d_bn(x=x, filters=(num_class+boudings*5), num_row=1, num_col=1, padding='same', strides=(1, 1), bn=False,activation=None)
    return x

def classifier_layers(features,num_class=19):
    x =layers.GlobalMaxPool2D()(features)
    x=layers.Dense(x.shape[1],activation='relu')(x)
    x=layers.Dense(num_class,activation='softmax')(x)
    return x
def getrpnmodel(input_shape=(448, 448, 3),positiondetect=True):
    img_input = layers.Input(shape=input_shape)
    # inception=ResNet50V2(include_top=False)
    features = head_network(img_input)
    headnetmodel=models.Model(img_input,features)
    if(positiondetect==False):
        x_class = classifier_layers(features)
        allmodel = models.Model(img_input,x_class)
    elif(positiondetect==True):
        class_boundings= boungding_and_classlayer(features,num_class=configg.classnum,boudings=configg.B)
        allmodel = models.Model(img_input, class_boundings)
    return headnetmodel,allmodel
# headnet,allmodel=getrpnmodel(positiondetect=True)
# allmodel.summary()