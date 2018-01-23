import pandas as pd
import numpy as numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers import LSTM, LeakyReLU, CuDNNLSTM, CuDNNGRU
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import h5py
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import regularizers


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

with h5py.File(''.join(['data/allcoin2015to2017_wf.h5']), 'r') as hf:
    datas = hf['inputs'].value
    labels = hf['outputs'].value


nb_partitions = datas.shape[0]

step_size = datas.shape[2]
units= 50
second_units = 30
batch_size = 8
nb_features = datas.shape[3]
epochs = 50
output_size=1

output_file_name='allcoin2015to2017_WF_CNN_2'
#split training validation

for partition in range(nb_partitions):
    training_datas = datas[partition,:8000*4,:,:]
    validation_datas = datas[partition,8000*4:,:,:]
    training_labels = labels[partition,:8000*4,:,:]
    validation_labels = labels[partition,8000*4:,:,:]
    early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    csvlog = CSVLogger('result/'+output_file_name+'.csv', append=True)
    checkpoint = ModelCheckpoint('weights/'+output_file_name+'partition_'+str(partition)+'.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    cb_list = [early,csvlog,checkpoint]
    print training_datas.shape, validation_datas.shape, training_labels.shape, validation_labels.shape
    
    #build model
    model = Sequential()
    
    # 2 Layers
    model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=3, filters=8, kernel_size=20))
    model.add(Dropout(0.2))

    model.add(Conv1D( strides=4, filters=1, kernel_size=16))

    '''
    # 3 Layers
    model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=3, filters=8, kernel_size=8))
    #model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Conv1D(activation='relu', strides=2, filters=8, kernel_size=8))
    #model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Conv1D( strides=2, filters=nb_features, kernel_size=8))


    # 4 layers
    model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=2, filters=8, kernel_size=2))
    #model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Conv1D(activation='relu', strides=2, filters=8, kernel_size=2))
    #model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Conv1D(activation='relu', strides=2, filters=8, kernel_size=2))
    #model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Conv1D( strides=2, filters=nb_features, kernel_size=2))
    '''
    model.compile(loss='mse', optimizer='adam')
    model.fit(training_datas, training_labels, batch_size=batch_size,validation_data=(validation_datas,validation_labels), epochs = epochs, callbacks=cb_list, verbose=0)

# model.fit(datas,labels)
#model.save(output_file_name+'.h5')


