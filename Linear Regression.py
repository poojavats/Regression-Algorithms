#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("height-weight.csv")


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.isnull().sum()


# In[6]:


plt.scatter(df['Weight'],df['Height'])
plt.xlabel('Weight')
plt.ylabel('height')


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.isnull().sum()


# In[10]:


## Divide our dataset into dependent and independent Features
X=df[['Weight']] ##Independent Feature - This should be series
y=df['Height'] ## dependent feature- This should be column


# In[11]:


X.head()


# In[12]:


X.shape


# In[13]:


y.shape


# In[14]:


## Further divide the dataset into tarin and test


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# In[17]:


X_train.shape,y_train.shape


# In[18]:


X_test.shape,y_test.shape


# In[19]:


## so beacuse unit should be in same range or scale down
from sklearn.preprocessing import StandardScaler


# In[20]:


scaler=StandardScaler()


# In[21]:


## we have to transform X_train data- fit means- the z score formula- mean and std calulate do after that transform will apply the formula in every value 
X_train=scaler.fit_transform(X_train)


# In[22]:


X_test=scaler.transform(X_test)


# In[23]:


X_test


# In[24]:


plt.scatter(X_train,y_train)


# In[25]:


scaler.transform([[80]])


# In[26]:


## Model Training
from sklearn.linear_model import LinearRegression


# In[27]:


regressor= LinearRegression()


# In[28]:


## Training the train Data
regressor.fit(X_train,y_train)


# In[29]:


#theta 0
regressor.intercept_


# In[31]:


# Theta 1
regressor.coef_


# In[32]:


plt.scatter(X_train,y_train)


# In[34]:


plt.scatter(X_train,y_train)
plt.scatter(X_train,regressor.predict(X_train))


# In[37]:


plt.scatter(X_train,y_train)
plt.plot(X_train,regressor.predict(X_train),color='r')


# # Prediction of Train_data
# 1. Predicted Height Output= intercept+coef_(weights)
# 2. y_pred_train=157.5+17.03(X_train)
# 
# # Prediction of Test_data
# 1. Predicted Height Output= intercept+coef_(weights)
# 2. y_pred_test=157.5+17.03(X_test)

# In[39]:


## Prediction for test data
Y_pred_test= regressor.predict(X_test)


# In[40]:


Y_pred_test


# In[42]:


y_test


# # Performance Metrics MAE,MSE,RMSE

# In[43]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[46]:


mse= mean_squared_error(y_test,Y_pred_test)
mae=mean_absolute_error(y_test,Y_pred_test)
rmse=np.sqrt(mse)
print(rmse)
print(mse)
print(mae)


# # Accuracy- R squared and Adjusted R squared
# # Formula
# # R^2=1-SSR/SST
# # R^2= Accuracy of the model
# # SSR= Sum of Squared of Residual
# # SST= Total Sum of Sqaures

# In[52]:


from sklearn.metrics import r2_score


# In[53]:


score= r2_score(y_test,Y_pred_test)
score


# In[55]:


## Display the Adjusted r square
1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)


# In[56]:


scaler


# In[57]:


regressor


# In[ ]:




