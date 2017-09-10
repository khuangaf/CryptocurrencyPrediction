
# coding: utf-8

# In[1]:


from keras import applications
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.callbacks import CSVLogger
import tensorflow as tf
from scipy.ndimage import imread
import numpy as np
import random
from keras.layers import LSTM
from keras import backend as K
import keras
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras import optimizers
import h5py
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# In[4]:


with h5py.File(''.join(['bitcoin2012_2017.h5']), 'r') as hf:
    datas = hf['datas'].value
    labels = hf['labels'].value
    # next_price = hf['next_price'].value
nb_samples = datas.shape[0]
nb_samples





# In[15]:


epochs = 50
batch_size = 15
step_size = 5
nb_validation_samples = int(0.3*nb_samples)
nb_training_samples = nb_samples - nb_validation_samples


# In[16]:


scaler = MinMaxScaler(feature_range=(0, 1))

training_datas = scaler.fit_transform(datas[:nb_training_samples])
validation_datas = scaler.fit_transform(datas[-nb_validation_samples:])

training_labels = labels[:nb_training_samples]
validation_labels = labels[-nb_validation_samples:]

training_next_price = training_datas[:,-1]
validation_next_price = validation_datas[:,-1]

training_datas = training_datas[:,:-1]
validation_datas = validation_datas[:,:-1]

training_datas = training_datas.reshape(nb_training_samples, step_size,1)
validation_datas = validation_datas.reshape(nb_validation_samples, step_size,1)

# print validation_datas.shape, training_datas.shape, validation_labels.shape


# In[17]:


model = Sequential()
model.add(LSTM(10
    , input_shape=(step_size,1),
    
    return_sequences=False))
model.add(Dropout(0.2))


model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()
# Adam = optimizers.Adam(lr=0.0001)
model.compile(loss='mse', optimizer='adam')
model.fit(training_datas, training_next_price, batch_size=batch_size,validation_data=(validation_datas,validation_next_price), epochs = epochs, callbacks=[CSVLogger('1layer.csv', append=True), ModelCheckpoint('weights-improvement-{epoch:02d}-{val_loss:.5f}.hdf5', monitor='val_loss', verbose=1,mode='min')])
model.save('1layer.h5')

# In[ ]:





# In[ ]:




