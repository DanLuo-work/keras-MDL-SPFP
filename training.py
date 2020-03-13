import pandas as pd
import datetime
import numpy as np
import os
from models.multiscale_multitask_STResNet import stresnet
from libs.utils import generate_x_y
import keras.backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping



def my_own_loss_function1(y_true, y_pred):
    return 10*K.mean(abs(y_true - y_pred)) + 0.0000005*K.mean(abs(K.clip(y_true,0.0001, 1)-K.clip(y_pred,0.0001, 1))/K.clip(y_true,0.0001, 1))*K.mean(abs(K.clip(y_true,0.0001, 1)-K.clip(y_pred,0.0001, 1))/K.clip(y_true,0.0001, 1))


def my_own_loss_function2(y_true, y_pred):
    return 20*K.mean(abs(y_true - y_pred)) + 0.0000005*K.mean(abs(K.clip(y_true,0.0001, 1)-K.clip(y_pred,0.0001, 1))/K.clip(y_true,0.0001, 1))*K.mean(abs(K.clip(y_true,0.0001, 1)-K.clip(y_pred,0.0001, 1))/K.clip(y_true,0.0001, 1))

def my_own_loss_function3(y_true, y_pred):
    return 10*K.mean(abs(y_true - y_pred)) + 0.00000005*K.mean(abs(K.clip(y_true,0.0001, 1)-K.clip(y_pred,0.0001, 1))/K.clip(y_true,0.0001, 1))*K.mean(abs(K.clip(y_true,0.0001, 1)-K.clip(y_pred,0.0001, 1))/K.clip(y_true,0.0001, 1))


def scheduler(epoch):
    # 每隔50个epoch，学习率减小为原来的1/2
    if epoch % 7 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)


A = np.load('tensor1_2_4_part_0.npz')
B1 = np.load('tensor3_week_day_part_0.npz')
B2 = np.load('tensor3_hour_part_0.npz')

t1_week = A['t1_week']
t1_day = A['t1_day']
t1_hour = A['t1_hour']
t1_target = A['t1_target'] 

t2_week = A['t2_week']
t2_day = A['t2_day']
t2_hour = A['t2_hour']
t2_target = A['t2_target'] 

t4_week = A['t4_week']
t4_day = A['t4_day']
t4_hour = A['t4_hour']
t4_target = A['t4_target'] 

t3_week = B1['t3_week']
t3_day = B1['t3_day']
t3_hour = B2['t3_hour']

length = t1_week.shape[0]
nb_flows = 2
L = 11
S = 267*2

len_seq1 = 1
len_seq2 = 1
len_seq3 = 3

x1_week = np.zeros([length, L, S, nb_flows*len_seq1])
x1_day = np.zeros([length, L, S, nb_flows*len_seq2])
x1_hour = np.zeros([length, L, S, nb_flows*len_seq3])
x1_target = np.squeeze(t1_target, axis=1)

x2_week = np.zeros([length, S, L*len_seq1])
x2_day = np.zeros([length, S, L*len_seq2])
x2_hour = np.zeros([length, S, L*len_seq3])
x2_target = np.zeros([length, S, L])

x3_week = np.zeros([length, L, S, L*len_seq1])
x3_day = np.zeros([length, L, S, L*len_seq2])
x3_hour = np.zeros([length, L, S, L*len_seq3])

x4_week = np.zeros([length, S, L*len_seq1])
x4_day = np.zeros([length, S, L*len_seq2])
x4_hour = np.zeros([length, S, L*len_seq3])
x4_target = np.zeros([length, S, L])

# tensor1转换
for i in range(length):
    for j in range(len_seq1):
        for k in range(nb_flows):
            x1_week[i, :, :, j*nb_flows+k] = t1_week[i, j, :, :, k]
for i in range(length):
    for j in range(len_seq2):
        for k in range(nb_flows):
            x1_day[i, :, :, j*nb_flows+k] = t1_day[i, j, :, :, k]
for i in range(length):
    for j in range(len_seq3):
        for k in range(nb_flows):
            x1_hour[i, :, :, j*nb_flows+k] = t1_hour[i, j, :, :, k]

# tensor2转换
for i in range(length):
    for j in range(len_seq1):
        for k in range(L):
            x2_week[i, :, j*L+k] = t2_week[i, j, k, :]
for i in range(length):
    for j in range(len_seq2):
        for k in range(L):
            x2_day[i, :, j*L+k] = t2_day[i, j, k, :]
for i in range(length):
    for j in range(len_seq3):
        for k in range(L):
            x2_hour[i, :, j*L+k] = t2_hour[i, j, k, :]
for i in range(length):
    for j in range(len_seq1):
        for k in range(L):
            x2_target[i, :, j*L+k] = t2_target[i, j, k, :]

# tensor3转换
for i in range(length):
    for j in range(len_seq1):
        for k in range(L):
            x3_week[i, :, :, j*L+k] = t3_week[i, j, :, k, :]
for i in range(length):
    for j in range(len_seq2):
        for k in range(L):
            x3_day[i, :, :, j*L+k] = t3_day[i, j, :, k, :]
for i in range(length):
    for j in range(len_seq3):
        for k in range(L):
            x3_hour[i, :, :, j*L+k] = t3_hour[i, j, :, k, :]

# tensor4转换
for i in range(length):
    for j in range(len_seq1):
        for k in range(L):
            x4_week[i, :, j*L+k] = t4_week[i, j, k, :]
for i in range(length):
    for j in range(len_seq2):
        for k in range(L):
            x4_day[i, :, j*L+k] = t4_day[i, j, k, :]
for i in range(length):
    for j in range(len_seq3):
        for k in range(L):
            x4_hour[i, :, j*L+k] = t4_hour[i, j, k, :]
for i in range(length):
    for j in range(len_seq1):
        for k in range(L):
            x4_target[i, :, j*L+k] = t4_target[i, j, k, :]


A = np.load('tensor1_2_3_4_part_0_transfer.npz')
A1 = np.load('tensor1_2_3_4_part_1_transfer.npz')
A2 = np.load('tensor1_2_3_4_part_2_transfer.npz')
A3 = np.load('tensor1_2_3_4_part_3_transfer.npz')
A4 = np.load('tensor1_2_3_4_part_4_transfer.npz')
A6 = np.load('tensor1_2_3_4_part_6_transfer.npz')

t1_week = np.concatenate((A['t1_week'],A1['t1_week'],A2['t1_week'],A3['t1_week'],A4['t1_week'],A6['t1_week']), axis=0) 
t1_day = np.concatenate((A['t1_day'],A1['t1_day'],A2['t1_day'],A3['t1_day'],A4['t1_day'],A6['t1_day']), axis=0)
t1_hour = np.concatenate((A['t1_hour'],A1['t1_hour'],A2['t1_hour'],A3['t1_hour'],A4['t1_hour'],A6['t1_hour']), axis=0)
t1_target = np.concatenate((A['t1_target'] ,A1['t1_target'],A2['t1_target'],A3['t1_target'],A4['t1_target'],A6['t1_target']), axis=0)

t2_week = np.concatenate((A['t2_week'],A1['t2_week'],A2['t2_week'],A3['t2_week'],A4['t2_week'],A6['t2_week']), axis=0) 
t2_day = np.concatenate((A['t2_day'],A1['t2_day'],A2['t2_day'],A3['t2_day'],A4['t2_day'],A6['t2_day']), axis=0)
t2_hour = np.concatenate((A['t2_hour'],A1['t2_hour'],A2['t2_hour'],A3['t2_hour'],A4['t2_hour'],A6['t2_hour']), axis=0)
t2_target = np.concatenate((A['t2_target'] ,A1['t2_target'],A2['t2_target'],A3['t2_target'],A4['t2_target'],A6['t2_target']), axis=0)

t3_week = np.concatenate((A['t3_week'],A1['t3_week'],A2['t3_week'],A3['t3_week'],A4['t3_week'],A6['t3_week']), axis=0) 
t3_day = np.concatenate((A['t3_day'],A1['t3_day'],A2['t3_day'],A3['t3_day'],A4['t3_day'],A6['t3_day']), axis=0)
t3_hour = np.concatenate((A['t3_hour'],A1['t3_hour'],A2['t3_hour'],A3['t3_hour'],A4['t3_hour'],A6['t3_hour']), axis=0)

 
t4_week = np.concatenate((A['t4_week'],A1['t4_week'],A2['t4_week'],A3['t4_week'],A4['t4_week'],A6['t4_week']), axis=0) 
t4_day = np.concatenate((A['t4_day'],A1['t4_day'],A2['t4_day'],A3['t4_day'],A4['t4_day'],A6['t4_day']), axis=0)
t4_hour = np.concatenate((A['t4_hour'],A1['t4_hour'],A2['t4_hour'],A3['t4_hour'],A4['t4_hour'],A6['t4_hour']), axis=0)
t4_target = np.concatenate((A['t4_target'] ,A1['t4_target'],A2['t4_target'],A3['t4_target'],A4['t4_target'],A6['t4_target']), axis=0)

#t1_week = A['t1_week'] 
#t1_day = A['t1_day']
#t1_hour = A['t1_hour']
#t1_target = A['t1_target']

#t2_week = A['t2_week'] 
#t2_day = A['t2_day']
#t2_hour = A['t2_hour']
#t2_target = A['t2_target']

#t3_week = A['t3_week'] 
#t3_day = A['t3_day']
#t3_hour = A['t3_hour']

#t4_week = A['t4_week'] 
#t4_day = A['t4_day']
#t4_hour = A['t4_hour']
#t4_target = A['t4_target']

model = stresnet(c_conf=(3, 11, 81), p_conf=(1, 11, 81), t_conf=(1, 11, 81),
                 c1_conf=(3, 11, 11, 81), p1_conf=(1, 11, 11, 81), t1_conf=(1, 11, 11, 81),
                 c2_conf=(3, 11, 81, 2), p2_conf=(1, 11, 81, 2), t2_conf=(1, 11, 81, 2),
                 external_dim1=11, external_dim2=2, external_dim3=2,
                 nb_task1_residual_unit=1, nb_task2_residual_unit=1, nb_task3_1_residual_unit=1, nb_task3_2_residual_unit=1)

model.compile(optimizer='adam', 
              loss={'task1': my_own_loss_function1, 'task2': my_own_loss_function2, 'task3': my_own_loss_function3},
              loss_weights={'task1': 1, 'task2': 1, 'task3': 1},
              metrics=['accuracy'])
model.summary()

#model.load_weights('log/1-1-1-1-128-64-kernel/49-0.43373208.hdf5')
# checkpoint
filepath = "log/3-3-3-3-64-kernel/{epoch:02d}-{loss:.8f}.hdf5"
# 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
reduce_lr = LearningRateScheduler(scheduler)
early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=0,  mode='min')
callbacks_list = [checkpoint, reduce_lr, early_stopping]
K.set_value(model.optimizer.lr, 0.001)
#顺序：趟次张量，车内人数张量，换乘张量，上下车人数张量 and hour->day->week
history = model.fit([t4_hour, t4_day, t4_week, t2_hour, t2_day, t2_week,
                     t3_hour, t3_day, t3_week, t1_hour, t1_day, t1_week], 
                    [t4_target, t2_target, t1_target], 
                    batch_size=128, epochs=52, callbacks=callbacks_list)
# print('-------------')




