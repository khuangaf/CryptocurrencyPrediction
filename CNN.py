import pandas as pd
import numpy as numpy
	
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.callbacks import CSVLogger, ModelCheckpoint
import h5py
import os


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


with h5py.File(''.join(['bitcoin2012_2017_256_16.h5']), 'r') as hf:
    datas = hf['inputs'].value
    labels = hf['outputs'].value


output_file_name='bitcoin2012_2017_256_16_CNN'

step_size = datas.shape[1]
batch_size= 8
nb_features = datas.shape[2]
epochs = 100

#split training validation
training_size = int(0.8* datas.shape[0])
training_datas = datas[:training_size,:]
training_labels = labels[:training_size,:]
validation_datas = datas[training_size:,:]
validation_labels = labels[training_size:,:]

#build model
model = Sequential()
model.add(Conv1D(activation="relu", input_shape=(step_size, nb_features), strides=3	, filters=8, kernel_size=8))
model.add(Dropout(0.25))
model.add(Conv1D(activation="relu", strides=2, filters=8, kernel_size=8))
model.add(Dropout(0.25))
model.add(Conv1D( strides=2, filters=4, kernel_size=8))

model.compile(loss='mse', optimizer='adam')
model.fit(training_datas, training_labels,verbose=0, batch_size=batch_size,validation_data=(validation_datas,validation_labels), epochs = epochs, callbacks=[CSVLogger(output_file_name+'.csv', append=True),ModelCheckpoint('weights/'+output_file_name+'weights-improvement-{epoch:02d}-{val_loss:.5f}.hdf5', monitor='val_loss', verbose=1,mode='min')])

# model.fit(datas,labels)
model.save(output_file_name+'.h5')


