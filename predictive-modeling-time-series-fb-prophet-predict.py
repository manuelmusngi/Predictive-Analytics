#!/usr/bin/env python
# coding: utf-8

# ## Time Series Prediction with FB Prophet
# - This is a price prediction project using FB Prophet, an open source tool used in Time Series Forecasting.
# - FB Prophet is known for its accuracy and forecasting simplicity. 
# - A procedure for forecasting time series data based on an additive model where non-linear trends are  fit with 
#   yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong 
#   seasonal effects and several # seasons of historical data.
# - It is an additive regression model with a piecewise linear or logistic growth curve trend. It includes a yearly 
#   seasonal component modeled using Fourier series and a weekly seasonal component modeled using dummy variables.

# ## Import Libraries

# In[ ]:


# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import random
import seaborn as sns
from fbprophet import Prophet

import warnings
warnings.simplefilter("ignore")


# ## Import Data

# In[ ]:


avocado_df = pd.read_csv('../input/avocado-prices-2020/avocado-updated-2020.csv')


# ## Exploratory Data Analysis 

# In[ ]:


avocado_df.head()


# In[ ]:


# View the last elements in the training dataset
avocado_df.tail(10)


# In[ ]:


avocado_df.describe()


# In[ ]:


avocado_df.info()


# In[ ]:


# inquire if the data contains null elements in the dataset
avocado_df.isnull().sum()


# In[ ]:


# sort values based on Date column
avocado_df = avocado_df.sort_values('date')


# In[ ]:


# Plot date and average price
plt.figure(figsize = (15,10))
plt.plot(avocado_df['date'], avocado_df['average_price'])


# In[ ]:


# Plot distribution of the average price
plt.figure(figsize=(15,6))
sns.distplot(avocado_df['average_price'], color='green');


# In[ ]:


# Plot a violin plot of the average price vs. avocado type
sns.violinplot(x='type', y='average_price', data=avocado_df)


# In[ ]:


# Bar Chart to indicate the number of regions 

sns.set(font_scale=0.7) 
plt.figure(figsize=[15,10])
sns.countplot(x = 'geography', data = avocado_df)
plt.xticks(rotation = 45)


# In[ ]:


# Bar Chart to indicate the count in every year
sns.set(font_scale=1.0) 
plt.figure(figsize=[10,5])
sns.countplot(x = 'year', data = avocado_df)
plt.xticks(rotation = 45)


# In[ ]:


# Create dataframe for FB Prophet
avocado_prophet_df = avocado_df[['date', 'average_price']] 


# In[ ]:


avocado_prophet_df


# In[ ]:


# Rename columns required in FB Prophet
avocado_prophet_df = avocado_prophet_df.rename(columns={'date':'ds', 'average_price':'y'})


# In[ ]:


avocado_prophet_df


# ## Modeling and Prediction

# In[ ]:


m = Prophet()
m.fit(avocado_prophet_df)


# In[ ]:


# Apply Predict
future = m.make_future_dataframe(periods=365)
prediction = m.predict(future)


# In[ ]:


prediction


# In[ ]:


figure = m.plot(prediction, xlabel='Date', ylabel='Price', figsize=(20,10))


# In[ ]:


# Visualize prediction in certain periods
figure2 =  m.plot_components(prediction, figsize = (20,10))

