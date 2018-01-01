
# coding: utf-8

# In[1]:


from keras import applications
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Activation, Reshape, LeakyReLU
from keras.callbacks import CSVLogger
import tensorflow as tf
from scipy.ndimage import imread
import numpy as np
import random
from keras.layers import LSTM, CuDNNLSTM
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
import h5py
from keras import regularizers
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[2]:


with h5py.File(''.join(['bitcoin2015to2017_close.h5']), 'r') as hf:
    datas = hf['inputs'].value
    labels = hf['outputs'].value
    input_times = hf['input_times'].value
    output_times = hf['output_times'].value
    original_inputs = hf['original_inputs'].value
    original_outputs = hf['original_outputs'].value
    original_datas = hf['original_datas'].value


# In[3]:


scaler=MinMaxScaler()
#split training validation
training_size = int(0.8* datas.shape[0])
training_datas = datas[:training_size,:,:]
training_labels = labels[:training_size,:,0]
validation_datas = datas[training_size:,:,:]
validation_labels = labels[training_size:,:,0]
validation_original_outputs = original_outputs[training_size:,:,:]
validation_original_inputs = original_inputs[training_size:,:,:]
validation_input_times = input_times[training_size:,:,:]
validation_output_times = output_times[training_size:,:,:]


# In[4]:


validation_original_inputs.shape


# In[5]:


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


# In[6]:


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# In[7]:


ground_true = np.append(validation_original_inputs,validation_original_outputs, axis=1)
ground_true_times = np.append(validation_input_times,validation_output_times, axis=1)
print ground_true_times.shape
print ground_true.shape


# In[8]:


step_size = datas.shape[1]
batch_size= 8
nb_features = datas.shape[2]
epochs = 30

output_size=16
units= 50


# In[9]:


training_datas.shape


# In[10]:


def fit_lstm(reg):
    global training_datas, training_labels, batch_size, epochs,step_size,nb_features, units
    model = Sequential()
    model.add(CuDNNLSTM(units=units, bias_regularizer=reg, input_shape=(step_size,nb_features),return_sequences=False))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(output_size))
    model.add(LeakyReLU())
    model.compile(loss='mse', optimizer='adam')
    model.fit(training_datas, training_labels, batch_size=batch_size, epochs = epochs, verbose=0)
    return model


# In[11]:


def experiment(validation_datas,validation_labels,original_datas,ground_true,ground_true_times,validation_original_outputs, validation_output_times, nb_repeat, reg):
    error_scores = list()
    #get only the close data
    ground_true = ground_true[:,:,0].reshape(-1)
    ground_true_times = ground_true_times.reshape(-1)
    ground_true_times = pd.to_datetime(ground_true_times, unit='s')
    validation_output_times = pd.to_datetime(validation_output_times.reshape(-1), unit='s')
    for i in range(nb_repeat):
        model = fit_lstm(reg)
        predicted = model.predict(validation_datas)
        predicted_inverted = []
        scaler.fit(original_datas[:,0].reshape(-1,1))
        predicted_inverted.append(scaler.inverse_transform(predicted))
        # since we are appending in the first dimension
        predicted_inverted = np.array(predicted_inverted)[0,:,:].reshape(-1)
        error_scores.append(mean_squared_error(validation_original_outputs[:,:,0].reshape(-1),predicted_inverted))
    return error_scores


# In[12]:


# iterate over different reglurizer
regs = [regularizers.l1(0),regularizers.l1(0.1), regularizers.l1(0.01), regularizers.l1(0.001), regularizers.l1(0.0001),regularizers.l2(0.1), regularizers.l2(0.01), regularizers.l2(0.001), regularizers.l2(0.0001)]
nb_repeat = 30
results = pd.DataFrame()
for reg in regs:
    
    name = ('l1 %.4f,l2 %.4f' % (reg.l1, reg.l2))
    print "Training "+ str(name)
    results[name] = experiment(validation_datas,validation_labels,original_datas,ground_true,ground_true_times,validation_original_outputs, validation_output_times, nb_repeat,reg)
# print(results.describe())
# save boxplot
results.describe().to_csv('result/lstm_bias_reg.csv')
results.describe()


# In[ ]:


def fit_lstm(reg):
    global training_datas, training_labels, batch_size, epochs,step_size,nb_features, units
    model = Sequential()
    model.add(CuDNNLSTM(units=units, kernel_regularizer=reg, input_shape=(step_size,nb_features),return_sequences=False))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(output_size))
    model.add(LeakyReLU())
    model.compile(loss='mse', optimizer='adam')
    model.fit(training_datas, training_labels, batch_size=batch_size, epochs = epochs, verbose=0)
    return model


# In[ ]:


# iterate over different reglurizer
regs = [regularizers.l1(0),regularizers.l1(0.1), regularizers.l1(0.01), regularizers.l1(0.001), regularizers.l1(0.0001),regularizers.l2(0.1), regularizers.l2(0.01), regularizers.l2(0.001), regularizers.l2(0.0001)]
nb_repeat = 30
results = pd.DataFrame()
for reg in regs:
    
    name = ('l1 %.4f,l2 %.4f' % (reg.l1, reg.l2))
    print "Training "+ str(name)
    results[name] = experiment(validation_datas,validation_labels,original_datas,ground_true,ground_true_times,validation_original_outputs, validation_output_times, nb_repeat,reg)

results.describe().to_csv('result/lstm_kernel_reg.csv')
results.describe()


# In[ ]:


def fit_lstm(reg):
    global training_datas, training_labels, batch_size, epochs,step_size,nb_features, units
    model = Sequential()
    model.add(CuDNNLSTM(units=units, activity_regularizer=reg, input_shape=(step_size,nb_features),return_sequences=False))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(output_size))
    model.add(LeakyReLU())
    model.compile(loss='mse', optimizer='adam')
    model.fit(training_datas, training_labels, batch_size=batch_size, epochs = epochs, verbose=0)
    return model


# In[ ]:


# iterate over different reglurizer
regs = [regularizers.l1(0),regularizers.l1(0.1), regularizers.l1(0.01), regularizers.l1(0.001), regularizers.l1(0.0001),regularizers.l2(0.1), regularizers.l2(0.01), regularizers.l2(0.001), regularizers.l2(0.0001)]
nb_repeat = 30
results = pd.DataFrame()
for reg in regs:
    
    name = ('l1 %.4f,l2 %.4f' % (reg.l1, reg.l2))
    print "Training "+ str(name)
    results[name] = experiment(validation_datas,validation_labels,original_datas,ground_true,ground_true_times,validation_original_outputs, validation_output_times, nb_repeat,reg)

results.describe().to_csv('result/lstm_activity_reg.csv')
results.describe()


# In[ ]:


def fit_lstm(reg):
    global training_datas, training_labels, batch_size, epochs,step_size,nb_features, units
    model = Sequential()
    model.add(CuDNNLSTM(units=units, recurrent_regularizer=reg, input_shape=(step_size,nb_features),return_sequences=False))
    model.add(Activation('tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(output_size))
    model.add(LeakyReLU())
    model.compile(loss='mse', optimizer='adam')
    model.fit(training_datas, training_labels, batch_size=batch_size, epochs = epochs, verbose=0)
    return model


# In[ ]:


# iterate over different reglurizer
regs = [regularizers.l1(0),regularizers.l1(0.1), regularizers.l1(0.01), regularizers.l1(0.001), regularizers.l1(0.0001),regularizers.l2(0.1), regularizers.l2(0.01), regularizers.l2(0.001), regularizers.l2(0.0001)]
nb_repeat = 30
results = pd.DataFrame()
for reg in regs:
    
    name = ('l1 %.4f,l2 %.4f' % (reg.l1, reg.l2))
    print "Training "+ str(name)
    results[name] = experiment(validation_datas,validation_labels,original_datas,ground_true,ground_true_times,validation_original_outputs, validation_output_times, nb_repeat,reg)

results.describe().to_csv('result/lstm_recurrent_reg.csv')
results.describe()

