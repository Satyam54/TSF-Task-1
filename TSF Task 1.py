#!/usr/bin/env python
# coding: utf-8

# # Python code for predicting Marks of a Student

# ## Made By : Satyam Choudhury

# ### Importing the Libraries

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


url="http://bit.ly/w-data"
s_data=pd.read_csv(url)
print("Data imported successfully")
s_data.head(10)


# In[38]:


s_data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours studied')
plt.ylabel('Pecentage score')
plt.show()


# In[39]:


X=s_data.iloc[:, :-1].values
y=s_data.iloc[:, 1].values


# In[40]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[41]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
print("Training Complete.")


# In[42]:


line = regressor.coef_*X+regressor.intercept_
plt.scatter(X,y)
plt.plot(X,line)
plt.show()


# In[43]:


print(X_test)
y_pred=regressor.predict(X_test)


# In[44]:


df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df


# In[55]:


hours = 9.25
#X_pred = regressor.predict([hours])
print("No of Hours = {}".format(hours))
print("Predicted Score = ",regressor.predict([[hours]]))


# In[56]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 

