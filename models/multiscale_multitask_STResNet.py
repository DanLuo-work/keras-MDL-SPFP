#coding:=utf-8

from __future__ import print_function
import numpy as np
from keras.layers import (
    Input,
    Activation,
    Dense,
    Reshape,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    multiply,
    Multiply,
    Embedding,
    Flatten,
    Add,
    Concatenate
)
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
#from keras.utils.visualize_util import plot
from models.iLayer3 import iLayer
import keras.backend as K
import keras.layers as KL


# SE块--全局池化再重新赋权重，实现注意力机制
def se_block(block_input, num_filters, flag=False, ratio=8):  # Squeeze and excitation block
    if flag:
        pool1 = GlobalAveragePooling2D()(block_input)
    else:
        pool1 = GlobalAveragePooling1D()(block_input)
    flat = Reshape((1, num_filters))(pool1)
    dense1 = Dense(num_filters // ratio, activation='relu')(flat)
    dense2 = Dense(num_filters, activation='sigmoid')(dense1)
    scale = multiply([block_input, dense2])
    return scale


def _shortcut(input, residual):
    return Add()([input, residual])


# BN层+ReLU层+卷积层，整合成一个块, flag为是否用2D卷积
def _bn_relu_conv(nb_filter, bn=False, flag=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=-1)(input)
        activation = Activation('relu')(input)
        if flag:
            return Convolution2D(nb_filter=nb_filter, kernel_size=(3,3),  border_mode="same")(activation)
        else:
            return Convolution1D(nb_filter=nb_filter, filter_length=3, border_mode="same")(activation)
    return f


# 残差单元，包含两个BN_ReLU_Conv块，以及一个SE快
def _residual_unit(nb_filter, flag=False):
    def f(input):
        residual = _bn_relu_conv(nb_filter, flag=flag)(input)
        residual = _bn_relu_conv(nb_filter, flag=flag)(residual)
        se = se_block(residual, num_filters=nb_filter, flag=flag)
        return _shortcut(input, se)
    return f


# 残差网络--包含repetations个残差单元，并在残差单元之间插入SE块
def ResUnits(residual_unit, nb_filter, repetations=1, flag=False):
    def f(input):
        for i in range(repetations):
            input = residual_unit(nb_filter=nb_filter, flag=flag)(input)
            se = se_block(input, num_filters=nb_filter, flag=flag)
        return se
    return f


# ST-ResNet网络
# 266为单向站点个数，加上反方向--> 266*2
def stresnet(c_conf=(3, 11, 266*2), p_conf=(3, 11, 266*2), t_conf=(3, 11, 266*2),
             c1_conf=(3, 11, 11, 266*2), p1_conf=(3, 11, 11, 266*2), t1_conf=(3, 11, 11, 266*2),
             c2_conf=(3, 11, 266*2, 2), p2_conf=(3, 11, 266*2, 2), t2_conf=(3, 11, 266*2, 2),
             external_dim1=None, external_dim2=None, external_dim3=None,
             nb_task1_residual_unit=1, nb_task2_residual_unit=1, nb_task3_1_residual_unit=1, nb_task3_2_residual_unit=1):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    c_conf = (len_seq, lines, stations)  # 　车上人数张量　趟次信息张量
    c1_conf = (len_seq, lines, lines, stations)　　#　　车之间转移张量　
    c2_conf = (len_seq, lines, stations, inflow/outflow)　　#  上下车人数
    external_dim为外部信息维度
    '''
    # main input
    main_inputs = []
    main_outputs = []
    branch_outputs = []
    outputs = []
    nb_flows = 2
    nb_stations = 81
    nb_lines = 11
    # get mask matrix(534, 11) & mask matrxi II(534, 11, 2)
#    input0 = Input(shape=(nb_stations, nb_lines))
#    input00 = Input(shape=(nb_stations, nb_lines, nb_flows)) 
#    main_inputs.append(input0)
#    main_inputs.append(input00)
    # task 1: 输入为趟次信息张量，用c_conf配置信息
    # 针对C、P、T三种时间范围的数据进行卷积
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, lines, stations = conf
            input1 = Input(shape=(stations, lines * len_seq))
            main_inputs.append(input1)
            # Conv1
            task1_conv1 = Convolution1D(
                nb_filter=64,  filter_length=3, border_mode="same")(input1)
            # [nb_residual_unit] Residual Units
            task1_residual_output = ResUnits(_residual_unit, nb_filter=64,
                                       repetations=nb_task1_residual_unit)(task1_conv1)
            # Conv2
            task1_activation = Activation('relu')(task1_residual_output)
            task1_conv2 = Convolution1D(
                nb_filter=64, filter_length=3, border_mode="same")(task1_activation)
            # Multi-scale
            task1_conv3_1 = Convolution1D(
                nb_filter=nb_lines, filter_length=3, border_mode="same")(task1_conv2)
            task1_conv3_2 = Convolution1D(
                nb_filter=nb_lines, filter_length=5, border_mode="same")(task1_conv2)
            task1_conv3_3 = Convolution1D(
                nb_filter=nb_lines, filter_length=7, border_mode="same")(task1_conv2)
            task1_conv3 = Add()([task1_conv3_1, task1_conv3_2, task1_conv3_3])
            task1_reshape = Reshape((nb_stations, nb_lines, 1))(task1_conv3)
            outputs.append(task1_reshape)

    # task 2: 输入为车上人数张量，也用c_conf配置信息
    # 针对C、P、T三种时间范围的数据进行卷积
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, lines, stations = conf
            input2 = Input(shape=(stations, lines * len_seq))
            main_inputs.append(input2)
            # Conv1
            task2_conv1 = Convolution1D(
                nb_filter=64,  filter_length=3, border_mode="same")(input2)
            # [nb_residual_unit] Residual Units
            task2_residual_output = ResUnits(_residual_unit, nb_filter=64,
                                       repetations=nb_task2_residual_unit)(task2_conv1)
            # Conv2
            task2_activation = Activation('relu')(task2_residual_output)
            task2_conv2 = Convolution1D(
                nb_filter=64, filter_length=3, border_mode="same")(task2_activation)
            # Multi-scale
            task2_conv3_1 = Convolution1D(
                nb_filter=nb_lines, filter_length=3, border_mode="same")(task2_conv2)
            task2_conv3_2 = Convolution1D(
                nb_filter=nb_lines, filter_length=5, border_mode="same")(task2_conv2)
            task2_conv3_3 = Convolution1D(
                nb_filter=nb_lines, filter_length=7, border_mode="same")(task2_conv2)
            task2_conv3 = Add()([task2_conv3_1, task2_conv3_2, task2_conv3_3])
            task2_reshape = Reshape((nb_stations, nb_lines, 1))(task2_conv3)
            outputs.append(task2_reshape)

    # task 3-branch1:输入为车间转移张量，用c1_conf配置信息
    # 针对C、P、T三种时间范围数据进行卷积
    for conf in [c1_conf, p1_conf, t1_conf]:
        if conf is not None:
            len_seq, lines, lines1, stations = conf
            input3 = Input(shape=(lines, stations, len_seq*lines1))
            main_inputs.append(input3)
            # Conv1
            task_3_1_conv1 = Convolution2D(
                    nb_filter=64, kernel_size=(5, 5), border_mode="same")(input3)
            # [nb_residual_unit] Residual Units
            task_3_1_residual_output = ResUnits(_residual_unit, nb_filter=64, repetations=nb_task3_1_residual_unit, flag=True)(task_3_1_conv1)
            # Conv2
            task_3_1_activation = Activation('relu')(task_3_1_residual_output)
            task_3_1_conv2 = Convolution2D(
                    nb_filter=64, kernel_size=(3, 3), border_mode="same")(task_3_1_activation)
            # Multi-scale
            task_3_1_conv3_1 = Convolution2D(
                nb_filter=nb_flows, kernel_size=(3, 3), border_mode="same")(task_3_1_conv2)
            task_3_1_conv3_2 = Convolution2D(
                nb_filter=nb_flows, kernel_size=(5, 5), border_mode="same")(task_3_1_conv2)
            task_3_1_conv3_3 = Convolution2D(
                nb_filter=nb_flows, kernel_size=(7, 7), border_mode="same")(task_3_1_conv2)
            task_3_1_conv3 = Add()([task_3_1_conv3_1, task_3_1_conv3_2, task_3_1_conv3_3])
            branch_outputs.append(task_3_1_conv3)

    # task 3-branch2:输入为车上下人数，用c2_conf配置信息
    # 针对C、P、T三种时间范围数据进行卷积
    for conf in [c2_conf, p2_conf, t2_conf]:
        if conf is not None:
            len_seq, lines, stations, flows = conf
            input4 = Input(shape=(lines, stations, len_seq*flows))

            main_inputs.append(input4)
            # Conv1
            task_3_2_conv1 = Convolution2D(
                    nb_filter=64, kernel_size=(5, 5), border_mode="same")(input4)
            # [nb_residual_unit] Residual Units
            task_3_2_residual_output = ResUnits(_residual_unit, nb_filter=64, repetations=nb_task3_2_residual_unit, flag=True)(task_3_2_conv1)
            # Conv2
            task_3_2_activation = Activation('relu')(task_3_2_residual_output)
            task_3_2_conv2 = Convolution2D(
                    nb_filter=64, kernel_size=(3, 3), border_mode="same")(task_3_2_activation)
            # Multi-scale
            task_3_2_conv3_1 = Convolution2D(
                nb_filter=nb_flows, kernel_size=(3, 3), border_mode="same")(task_3_2_conv2)
            task_3_2_conv3_2 = Convolution2D(
                nb_filter=nb_flows, kernel_size=(5, 5), border_mode="same")(task_3_2_conv2)
            task_3_2_conv3_3 = Convolution2D(
                nb_filter=nb_flows, kernel_size=(7, 7), border_mode="same")(task_3_2_conv2)
            task_3_2_conv3 = Add()([task_3_2_conv3_1, task_3_2_conv3_2, task_3_2_conv3_3])
            branch_outputs.append(task_3_2_conv3)

    # -----bridge 1-----将转移矩阵和上下车人数相加
    task3_output = Add()(branch_outputs)
    re_task3_output = Reshape((nb_stations, nb_lines, 2))(task3_output)
    outputs.append(re_task3_output)    
    # -----bridge 2-----将趟次张量，车内人数，上下车人数张量concatenate
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        # Bridge操作，即对数据进行简单地concatenate操作
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output)) 
        main_output = Concatenate()(new_outputs)

    # 对Bridge数据进行不同维度地卷积，实现3个不同任务的输出
    conv_task1 = Convolution2D(nb_filter=1, kernel_size=(3, 3), border_mode="same")(main_output)
    reshape_task1 = Reshape((nb_stations, nb_lines))(conv_task1)

    conv_task2 = Convolution2D(nb_filter=1, kernel_size=(3, 3), border_mode="same")(main_output)
    reshape_task2 = Reshape((nb_stations, nb_lines))(conv_task2)

    conv_task3 = Convolution2D(nb_filter=nb_flows, kernel_size=(3, 3), border_mode="same")(main_output)
    reshape_task3 = Reshape((nb_lines, nb_stations, nb_flows))(conv_task3)


    main_outputs.append(reshape_task1)
    main_outputs.append(reshape_task2)
    main_outputs.append(reshape_task3)

    # fusing node/edge with external component1
    # external_dim1 (81, 11, 11) & external_dim2 (2) & external_dim3 (2)
    # 将外部信息分别整合到task1数据,task2数据和task3数据中
    if external_dim1 is not None and external_dim1 > 0:
        # external input 外部信息输入
        external_input1 = Input(shape=(81, 11, external_dim1))
        main_inputs.append(external_input1)
        # convolution
        external_dim1_conv1 = Convolution2D(
                              nb_filter=64, kernel_size=(3, 3), border_mode="same")(external_input1)        
        #外部信息加入到task1数据中
        external1_h1 = Convolution2D(
                              nb_filter=1, kernel_size=(3, 3), border_mode="same")(external_dim1_conv1)               
        external1_activation1 = Activation('relu')(external1_h1)
        external1_output1 = Reshape((nb_stations, nb_lines))(external1_activation1)
        main_output1 = Add()([reshape_task1, external1_output1])
        main_outputs[0] = main_output1
        #外部信息加入到task2数据中
        external1_h2 = Convolution2D(
                              nb_filter=1, kernel_size=(3, 3), border_mode="same")(external_dim1_conv1) 
        external1_activation2 = Activation('relu')(external1_h2)
        external1_output2 = Reshape((nb_stations, nb_lines))(external1_activation2)
        main_output2 = Add()([reshape_task2, external1_output2])
        main_outputs[1] = main_output2
        #外部信息加入到task3数据中
        external1_h3 = Convolution2D(
                              nb_filter=2, kernel_size=(3, 3), border_mode="same")(external_dim1_conv1)
        external1_activation3 = Activation('relu')(external1_h3)
        external1_output3 = Reshape((nb_lines, nb_stations, nb_flows))(external1_activation3)
        main_output3 = Add()([reshape_task3, external1_output3])
        main_outputs[2] = main_output3
    else:
        print('external_dim:', external_dim1)

    # fusing node/edge with external component2
    # 将外部信息分别整合到task1数据,task2数据和task3数据中
    if external_dim2 is not None and external_dim2 > 0:
        # external input 外部信息输入
        external_input2 = Input(shape=(external_dim2,))
        main_inputs.append(external_input2)
        embedding2 = Embedding(31, 30, input_length=2)(external_input2)
        flatten_embedding2 = Flatten()(embedding2)
        #外部信息加入到task1数据中
        external2_h1 = Dense(output_dim=nb_lines * nb_stations)(flatten_embedding2)
        external2_activation1 = Activation('relu')(external2_h1)
        external2_output1 =Reshape((nb_stations, nb_lines))(external2_activation1)
        main_output1 = Add()([reshape_task1, external2_output1])
        main_outputs[0] = main_output1
        #外部信息加入到task2数据中
        external2_h2 = Dense(output_dim=nb_lines * nb_stations)(flatten_embedding2)
        external2_activation2 = Activation('relu')(external2_h2)
        external2_output2 = Reshape((nb_stations, nb_lines))(external2_activation2)
        main_output2 = Add()([reshape_task2, external2_output2])
        main_outputs[1] = main_output2
        #外部信息加入到task3数据中
        external2_h3 = Dense(output_dim=nb_flows * nb_stations * nb_lines)(flatten_embedding2)
        external2_activation3 = Activation('relu')(external2_h3)
        external2_output3 = Reshape((nb_lines, nb_stations, nb_flows))(external2_activation3)
        main_output3 = Add()([reshape_task3, external2_output3])
        main_outputs[2] = main_output3
    else:
        print('external_dim:', external_dim2)

    # fusing node/edge with external component3
    # 将外部信息分别整合到task1数据,task2数据和task3数据中
    if external_dim3 is not None and external_dim3 > 0:
        # external input 外部信息输入
        external_input3 = Input(shape=(external_dim3,))
        main_inputs.append(external_input3)
        embedding3 = Embedding(2, 30, input_length=2)(external_input3)
        flatten_embedding3 = Flatten()(embedding3)
        #外部信息加入到task1数据中
        external3_h1 = Dense(output_dim=nb_lines * nb_stations)(flatten_embedding3)
        external3_activation1 = Activation('relu')(external3_h1)
        external3_output1 = Reshape((nb_stations, nb_lines))(external3_activation1)
        main_output1 = Add(name='task1')([reshape_task1, external3_output1])
        main_outputs[0] = main_output1
        #外部信息加入到task2数据中
        external3_h2 = Dense(output_dim=nb_lines * nb_stations)(flatten_embedding3)
        external3_activation2 = Activation('relu')(external3_h2)
        external3_output2 = Reshape((nb_stations, nb_lines))(external3_activation2)
        main_output2 = Add(name='task2')([reshape_task2, external3_output2])
        main_outputs[1] = main_output2
        #外部信息加入到task3数据中
        external3_h3 = Dense(output_dim=nb_flows * nb_stations * nb_lines)(flatten_embedding3)
        external3_activation3 = Activation('relu')(external3_h3)
        external3_output3 = Reshape((nb_lines, nb_stations, nb_flows))(external3_activation3)
        main_output3 = Add(name='task3')([reshape_task3, external3_output3])
        main_outputs[2] = main_output3
    else:
        print('external_dim:', external_dim3)

    # 完成模型搭建 
    model = Model(main_inputs, main_outputs)
    return model
