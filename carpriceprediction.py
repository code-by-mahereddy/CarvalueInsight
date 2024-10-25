#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pickle


# In[128]:


import pandas as pd
data=pd.read_csv("C:\\Users\\rama\\Downloads\\fiat500.csv")
data


# In[129]:


data.info()


# In[130]:


data.head(94)


# In[131]:


data['km'].corr(data['price'])


# In[132]:


data['km'].cov(data['price'])


# In[133]:


data.describe()


# In[134]:


data['engine_power'].mean()


# In[135]:


data['engine_power'].mode()


# In[136]:


data['model'].unique()


# In[137]:


data.shape


# In[138]:


data['previous_owners'].unique()


# In[139]:


data.groupby(data['previous_owners']).count()


# In[140]:


data.groupby(data['model']).count()


# In[141]:


data


# In[142]:


#linear regression
df=data.drop(data.columns[[0,5,6,7]],axis=1)
df


# In[143]:


from sklearn import preprocessing  
label_encoder = preprocessing.LabelEncoder() 
 
df['model']= label_encoder.fit_transform(df['model']) 
df['model'].unique()



# In[144]:


y=df['price']
x=df.drop(['price'],axis=1)


# In[145]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)


# In[146]:


x_train.shape


# In[147]:


print(x_train)


# In[148]:


x_test.shape


# In[149]:


print(y_train)


# In[150]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x,y)


# In[152]:


reg.coef_


# In[153]:


reg.intercept_


# In[154]:


from sklearn.metrics import r2_score
r2_score(y_test,reg.predict(x_test))


# In[155]:


pred=reg.predict(x_test)
from matplotlib import pyplot as plt
print(pred)


# In[156]:


results=pd.DataFrame(columns=['price','predicted'])
results['price']=y_test
results['predicted']=pred
results=results.reset_index()
results['id']=results.index
results.head()


# In[157]:


import seaborn as sns
sns.lineplot(x='id',y='price',data=results.tail(30),color='green')
sns.lineplot(x='id',y='predicted',data=results.tail(30),color='red')


# In[158]:


sns.relplot(x='id',y='predicted',data=results.tail(30),kind='line')


# In[169]:


print(reg.predict(pd.DataFrame({'model': [1], 'engine_power': [51], 'age_in_days': [10], 'km': [3000]})))


# In[160]:


print(data)


# In[ ]:




