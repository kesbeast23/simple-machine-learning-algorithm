#!/usr/bin/env python
# coding: utf-8

# In[67]:


#importing all the necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plot                     #Data visualisation libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.model_selection import train_test_split
    
df = pd.read_csv (r"/home/tumo505/Documents/school/is assignment/FyntraCustomerData.csv.csv")  


# In[3]:


#print the dataset
df


# In[4]:


#replace all the '?' with 'None'
df.replace('?', None)


# In[24]:


#replace all the missing values with the most frequently appearing values in the dataset
#imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
#print(pd.DataFrame(imp.fit_transform(df),
#                   columns=df.columns,
 #                  index=df.index))
    
#Replacing missing values with the most frequencly occuring data in each column
df.Avg_Session_Length.fillna(df.Avg_Session_Length.mode()[0], inplace=True)
df.Time_on_App.fillna(df.Time_on_App.mode()[0], inplace=True)
df.Time_on_Website.fillna(df.Time_on_Website.mode()[0], inplace=True)
df.Length_of_Membership.fillna(df.Length_of_Membership.mode()[0], inplace=True)
df.Yearly_Amount_Spent.fillna(df.Yearly_Amount_Spent.mode()[0], inplace=True)


# In[47]:


#check if there are any missing values
df.isnull().values.any()


# In[50]:


#to get all the statistical info about the data
df.describe()


# In[51]:


#convert the columns to numeric
cols = ['Time_on_Website', 'Yearly_Amount_Spent']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)


# In[54]:


#create a joint plot to compare the two columns 
datafr =pd.DataFrame(df)
sns.jointplot(x="Time_on_Website", y="Yearly_Amount_Spent", data=df);

#there seems to be no correlation


# In[55]:


#convert the columns to numeric
cols = ['Time_on_App', 'Yearly_Amount_Spent']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)


# In[124]:


#compare the two columns using joitplot
datafr =pd.DataFrame(df)
sns.jointplot(x="Time_on_App", y="Yearly_Amount_Spent", data=df);

#there is some correlation between "yearly amount spent" and "time on app"


# In[59]:


#convert all the columns to numeric
cols = ['Avg_Session_Length', 'Time_on_App', 'Time_on_Website', 'Length_of_Membership', 'Yearly_Amount_Spent']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)


# In[60]:


#Explore types of relationships across the entire dataset using Pairplot
sns.pairplot(df)

#seems that the yealy amount spent has the most correlation with the length of membership


# In[61]:


#checking correlation between the variables in the dataset
df.corr()


# In[62]:


#just checking out the data
#Returns the first 5 rows of the dataframe
df.head()


# In[63]:


#checking out the data

#Get basic details about the dataset
df.info()


# In[64]:


df.columns


# In[66]:


#convert the columns to numeric and plot the linear model

#shows that length of membership is directly propotional to the yealy amount spent
cols = ['Length_of_Membership', 'Yearly_Amount_Spent']
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)

sns.lmplot(x="Length_of_Membership", y="Yearly_Amount_Spent", data=df);


# In[130]:


#all numeric columns
X=df[['Avg_Session_Length','Time_on_App','Time_on_Website','Length_of_Membership']]

#columns which needs to be predicted
y=df[['Yearly_Amount_Spent']]

#Tuple Unpacking
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

#if we dont use the 'random state', we will get a different result everytime we run the 'train_test_split'


# In[38]:


#Create an instance of LinearRegression() Model
lm = LinearRegression()

#Fit Data on lm
lm.fit(X_train,y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

#print out the intercepts
print (lm.intercept_)


# In[40]:


#predicting the test data
predictions = lm.predict(X_test)


# In[41]:


#checking some of the predictions
predictions[0:5]


# In[42]:


#comparing the 'actual data' against the 'predicted data'
plot.scatter(y_test, predictions)
plot.xlabel("Actual data")
plot.ylabel("Predicted data")


# In[45]:


#computing the Root Mean Squared Error(RMSE)
#RMSE is a quadratic scoring rule that also measures the average magnitude of the error. 
#It's the square root of the average
print('The RSME is ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




