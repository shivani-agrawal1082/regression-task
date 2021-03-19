#!/usr/bin/env python
# coding: utf-8

# # Task :Predict the percentage of an student based on the no. of study hours.
# 
# # Name: Shivani Agrawal

# In[37]:


#Importing required libraries
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Reading data from remote link
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")

data.head()


# In[13]:


data.info()


# In[15]:


data.describe()


# In[14]:


# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[16]:


#finding the pearsons correlation coefficient

data.corr(method='pearson')


# In[23]:


Hours=data["Hours"]
Scores=data["Scores"]
sns.displot(Hours)


# 
# # Linear Regresion

# In[21]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values 


# In[25]:


#spliting the data

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0) 


# In[26]:


#training data set

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 


# In[16]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[40]:


y_pred=regressor.predict(X_test)


# In[41]:


actual_predicted=pd.DataFrame({'Target':y_test,'Predictd':y_pred})
actual_predicted


# In[42]:


sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_pred))
plt.show()


# # what will be the predicted score if a student studies for 9.25hrs/day ?

# In[44]:


s=9.25
t=regressor.predict([[s]])
print(t)


# In[45]:


print("so, if a student studies for 9.25hrs a day then he/she gets",t,"% marks in the exam" )


# # model evaluation

# In[46]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# which shows that 94.5490 % of variation in y is explained by the regression model.
