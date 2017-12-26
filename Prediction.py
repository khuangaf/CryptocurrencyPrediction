
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
from keras.layers import Conv1D, MaxPooling1D
from keras import backend as K
import keras
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras import optimizers
import h5py
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
# import matplotlib

import matplotlib.pyplot as plt
plt.switch_backend('agg')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# In[2]:


with h5py.File(''.join(['bitcoin2012_2017_256_16.h5']), 'r') as hf:
    datas = hf['inputs'].value
    labels = hf['outputs'].value
    input_times = hf['input_times'].value
    output_times = hf['output_times'].value
    original_datas = hf['original_datas'].value
    original_outputs = hf['original_outputs'].value




# In[3]:




# In[4]:

# For CNN
scaler=MinMaxScaler()
#split training validation
training_size = int(0.8* datas.shape[0])
training_datas = datas[:training_size,:]
training_labels = labels[:training_size,:]
validation_datas = datas[training_size:,:]
validation_labels = labels[training_size:,:]
ground_true = original_outputs[training_size:,:]
# For LSTM

# nb_samples = datas.shape[0]
# nb_samples
# datetimes = np.load('datetime.npy')
# epochs = 50
# batch_size = 15
# step_size = 5
# nb_validation_samples = int(0.3*nb_samples)
# nb_training_samples = nb_samples - nb_validation_samples

# input_step_size = 50
# output_size = 30
# scaler = MinMaxScaler(feature_range=(0, 1))

# training_datas = scaler.fit_transform(datas[:nb_training_samples])
# validation_datas = scaler.fit_transform(datas[-nb_validation_samples:])

# training_labels = labels[:nb_training_samples]
# validation_labels = labels[-nb_validation_samples:]

# training_next_price = training_datas[:,-1]
# validation_next_price = validation_datas[:,-1]

# training_datas = training_datas[:,:-1]
# validation_datas = validation_datas[:,:-1]

# training_datas = training_datas.reshape(nb_training_samples, step_size,1)
# validation_datas = validation_datas.reshape(nb_validation_samples, step_size,1)



step_size = datas.shape[1]
batch_size= 8
nb_features = datas.shape[2]
epochs = 1

#build model
model = Sequential()
model.add(Conv1D(activation="relu", input_shape=(step_size, nb_features), strides=3	, filters=8, kernel_size=8))
model.add(Dropout(0.25))
model.add(Conv1D(activation="relu", strides=2, filters=8, kernel_size=8))
model.add(Dropout(0.25))
model.add(Conv1D( strides=2, filters=4, kernel_size=8))
model.load_weights('weights/bitcoin2012_2017_256_16_CNNweights-improvement-02-0.00011.hdf5')
model.compile(loss='mse', optimizer='adam')


# In[5]:


# model = Sequential()
# model.add(LSTM(10
#     , input_shape=(input_step_size,1),
    
#     return_sequences=False))
# model.add(Dropout(0.2))


# model.add(Dense(output_size))
# model.add(Activation('sigmoid'))

# model.load_weights('weights/bitcoin2012_2017_50_30_weights.hdf5')
# model.compile(loss='mse', optimizer='adam')
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaler.fit(datas.reshape(-1))
# predicted_inverted = scaler.inverse_transform(predicted)
# ground_true = scaler.inverse_transform(validation_next_price)
# In[6]:


predicted = model.predict(validation_datas)
predicted_inverted = []

# In[7]:
for i in range(original_datas.shape[1]):
	scaler.fit(original_datas[:,i].reshape(-1,1))
	predicted_inverted.append(scaler.inverse_transform(predicted[:,:,i]))

#get only the close data
ground_true = ground_true[:,:,0].reshape(-1)
output_times = output_times.reshape(-1)

predicted_inverted = np.array(predicted_inverted)[:,:,0].reshape(-1)



# In[8]:
print output_times.shape, ground_true.shape

plt.plot(output_times[-1000:],ground_true[-1000:])
plt.plot(output_times[-1000:],predicted_inverted[-1000:])

# In[ ]:


plt.show()


# In[ ]:




