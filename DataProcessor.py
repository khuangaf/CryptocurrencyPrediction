
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import h5py


# In[24]:


input_step_size = 50
output_size = 30
sliding_window = False
file_name= 'bitcoin2012_2017_50_30_prediction.h5' 


# In[19]:


df = pd.read_csv('data/bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv').dropna().tail(1000000)
df['Datetime'] = pd.to_datetime(df['Timestamp'],unit='s')
df.head()


# In[30]:


prices= df.loc[:,'Close'].values
times = df.loc[:,'Close'].values
prices.shape


# In[31]:


outputs = []
inputs = []
output_times = []
input_times = []
if sliding_window:
    for i in range(len(prices)-input_step_size-output_size):
        inputs.append(prices[i:i + input_step_size])
        input_times.append(times[i:i + input_step_size])
        outputs.append(prices[i + input_step_size: i + input_step_size+ output_size])
        output_times.append(times[i + input_step_size: i + input_step_size+ output_size])
else:
    for i in range(0,len(prices)-input_step_size-output_size, input_step_size):
        inputs.append(prices[i:i + input_step_size])
        input_times.append(times[i:i + input_step_size])
        outputs.append(prices[i + input_step_size: i + input_step_size+ output_size])
        output_times.append(times[i + input_step_size: i + input_step_size+ output_size])
inputs= np.array(inputs)
outputs= np.array(outputs)
output_times = np.array(output_times)
input_times = np.array(input_times)


# In[34]:


with h5py.File(file_name, 'w') as f:
    f.create_dataset("inputs", data = inputs)
    f.create_dataset('outputs', data = outputs)
    f.create_dataset("input_times", data = input_times)
    f.create_dataset('output_times', data = output_times)

