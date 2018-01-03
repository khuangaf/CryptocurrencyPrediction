from __future__ import print_function
from flask import Flask, render_template,request
import sys

from flask_socketio import send, emit
from os import environ
from flask import Flask
import json
import numpy as np
import os
import pandas as pd
import urllib2
from datetime import datetime
import calendar
from keras import applications
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Activation, Reshape, LeakyReLU
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
from sklearn.preprocessing import MinMaxScaler
import h5py
from flask import jsonify
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'


#app.run(environ.get('PORT'))



@app.route('/api/predict')
def api_predict():
	d = datetime.utcnow()
	unixtime = calendar.timegm(d.utctimetuple())

	unixtime = unixtime /100 *100
	past_unixtime = unixtime- 300*300
	url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start='+str(past_unixtime)+'&end=9999999999&period=300'
	openUrl = urllib2.urlopen(url)
	r = openUrl.read()
	openUrl.close()
	d = json.loads(r.decode())
	df = pd.DataFrame(d)
	original_columns=[u'close', u'date', u'high', u'low', u'open']
	new_columns = ['Close','Timestamp','High','Low','Open']
	df = df.loc[:,original_columns]
	df.columns = new_columns
	df = df.iloc[-256:,:]
	datas = df.Close
	with h5py.File(''.join(['bitcoin2015to2017_close.h5']), 'r') as hf:
    	original_datas = hf['original_datas'].value

	scaler = MinMaxScaler()
	scaler.fit(original_datas[:,0].reshape(-1,1))
	datas= scaler.transform(datas.reshape(-1,1))
	datas=datas[None,:,:]
	#datas.shape = [1,256,1]
	step_size = datas.shape[1]
	batch_size= 8
	nb_features = datas.shape[2]
	epochs = 1
	output_size=16
	units= 50
	second_units=30
	model = Sequential()
	model.add(LSTM(units=units, activation=None, input_shape=(step_size,nb_features),return_sequences=False))
	model.add(Activation('tanh'))
	model.add(Dropout(0.2))
	model.add(Dense(output_size))
	model.add(LeakyReLU())
	model.load_weights('weights/bitcoin2015to2017_close_LSTM_1_tanh_leaky_-44-0.00004.hdf5')
	model.compile(loss='mse', optimizer='adam')
	predicted = model.predict(datas)
	predicted_inverted = scaler.inverse_transform(predicted)
	predicted_inverted
	return return jsonify(predicted_inverted)
@app.route('/')
def index():
	return render_template("index.html")
# app.run(port="8080")
if __name__ == '__main__':
	HOST = environ.get('SERVER_HOST', 'localhost')
	try:
		PORT = int(environ.get('PORT','5009'))
	except ValueError:
		PORT = 5009
   app.run(HOST, PORT)
	# socketio.run(app, port =  PORT, host= '0.0.0.0')