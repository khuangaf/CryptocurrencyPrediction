
# coding: utf-8

# In[1]:


import urllib2
import json
import pandas as pd
import h5py


# In[6]:


res = urllib2.urlopen("https://api.coindesk.com/v1/bpi/historical/close.json?start=2010-09-01&end=2017-09-01").read()
res = json.loads(res)


# In[7]:


prices = pd.DataFrame(res).loc[:,'bpi'].values[:-2]
prices


# In[8]:
input_step_size = 50
output_size = 100

labels = []
datas = []
next_price = []
step_size = 6
for i in range(len(prices)-step_size):
    datas.append(prices[i:i+step_size])
    if prices[i+step_size] >= prices[i+(step_size-1)]:
        labels.append(1)
    else:
        labels.append(0)
    


# In[9]:


with h5py.File(''.join(['bitcoin2013_2019.h5']), 'w') as f:
    f.create_dataset("datas", data = datas)
    f.create_dataset('labels', data = labels)



# In[20]:


df = pd.read_csv('data/bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv').dropna().tail(1000000)


# In[21]:


df


# In[27]:


prices= df.loc[:,'Close'].values
prices.shape


# In[28]:


labels = []
datas = []
step_size = 6
for i in range(len(prices)-step_size):
    datas.append(prices[i:i+step_size])
    if prices[i+step_size] >= prices[i+(step_size-1)]:
        labels.append(1)
    else:
        labels.append(0)
    


# In[29]:


with h5py.File(''.join(['bitcoin2012_2017_50_30.h5']), 'w') as f:
    f.create_dataset("datas", data = datas)
    f.create_dataset('labels', data = labels)


# In[ ]:




