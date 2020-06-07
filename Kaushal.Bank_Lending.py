#!/usr/bin/env python
# coding: utf-8

# #                   data model to predict the probability of default for XYZ org.

# ### IMPORTING Libraries

# In[1]:


import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import os


# ## Reading the data from source file 

# In[2]:


os.chdir(r"C:\Users\Videos\Imarticus")
data = pd.read_csv('XYZCorp_LendingData.txt',sep="\t",low_memory=False)


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.dtypes


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.describe(include=object)


# ## Exploratory Data Analysis

# ##### Checking NULL values and treating them

# In[9]:


data.isnull().sum()

A lot of columns contain Null values some we can treat some we can choose to ignore as majority of datapoints are null in them.
To start with we will drop the columns 
# In[10]:


null_cols = [x for x in data.count() < 855969*0.40]
data.drop(data.columns[null_cols], 1, inplace=True)


# In[11]:


data.shape


# We observe 20 columns are dropped which had more than 60% NAN values of the total datapoint, hence they would not have contributed towards prediction. For the remaining columns which have null values we will check and treat them using various methods

# #### Treating Null values

# In[12]:


#mths_since_last_delinq has 439812 null values lets check their various parameters
print (data.mths_since_last_delinq.min(), data.mths_since_last_delinq.max())
print(data.mths_since_last_delinq.mean())
print(data.mths_since_last_delinq.median())
print(data.mths_since_last_delinq.mode())


# In[13]:


# we will replace NA values with Median value of the column, after treatement we see zero Null Values
data.mths_since_last_delinq = data.mths_since_last_delinq.fillna(data.mths_since_last_delinq.median())
data.mths_since_last_delinq.isnull().sum()


# In[14]:


#emp_title- Lets check various parameters
data['emp_title'].describe()


# We observe a lot of unique values here more than 29000 which would be difficlut to visualize, Lets see graphically 
# probably we will take a call later in visualization to keep it or drop this feature.

# In[15]:


#revol_util- Lets check various parameters here
print (data.revol_util.min(), data.revol_util.max())
print(data.revol_util.mean())
print(data.revol_util.median())
print(data.revol_util.mode())


# In[16]:


#We will replace null values here with mean of this feature. 
data.revol_util = data.revol_util.fillna(data.revol_util.mean())
data.revol_util.isnull().sum()


# In[17]:


#last_pymnt_d- Since these are dates we will fill the null values with previous values
data["last_pymnt_d"].fillna( method ='ffill', inplace = True) 
#next_pymnt_d- Since these are dates we will fill the null values with previous values
data["next_pymnt_d"].fillna( method ='ffill', inplace = True) 


# In[18]:


#  -- tot_coll_amt  -- tot_cur_bal -- total_rev_hi_lim- Lets check features of all and treat NA with Mean
#tot_coll_amt
print (data.tot_coll_amt.min(), data.tot_coll_amt.max())
print(data.tot_coll_amt.mean())
print(data.tot_coll_amt.median())
print(data.tot_coll_amt.mode())
print('*********************')
print('*********************')

#tot_cur_bal
print (data.tot_cur_bal.min(), data.tot_cur_bal.max())
print(data.tot_cur_bal.mean())
print(data.tot_cur_bal.median())
print(data.tot_cur_bal.mode())
print('*********************')
print('*********************')
#total_rev_hi_lim
print (data.total_rev_hi_lim.min(), data.total_rev_hi_lim.max())
print(data.total_rev_hi_lim.mean())
print(data.total_rev_hi_lim.median())
print(data.total_rev_hi_lim.mode())

data.tot_coll_amt = data.tot_coll_amt.fillna(data.tot_coll_amt.mean())
data.tot_cur_bal = data.tot_cur_bal.fillna(data.tot_cur_bal.mean())
data.total_rev_hi_lim = data.total_rev_hi_lim.fillna(data.total_rev_hi_lim.mean())


# In[19]:


#last_credit_pull_d- replace null with previous datapoint
data["last_credit_pull_d"].fillna( method ='ffill', inplace = True) 


# In[20]:


#collections_12_mths_ex_med- Lets check various features
print (data.collections_12_mths_ex_med.min(), data.collections_12_mths_ex_med.max())
print(data.collections_12_mths_ex_med.mean())
print(data.collections_12_mths_ex_med.median())
print(data.collections_12_mths_ex_med.mode())


# In[21]:


#replacing NA with Median
data.collections_12_mths_ex_med = data.collections_12_mths_ex_med.fillna(data.collections_12_mths_ex_med.median())


# In[22]:


#title
data['title'].describe()


# In[23]:


#60991 unique values this may not convey much information so will later take call to discard or keept it
data["title"].fillna( method ='ffill', inplace = True)


# In[24]:


#next_pymnt_d
data["next_pymnt_d"].value_counts()


# In[25]:


data["next_pymnt_d"].describe()


# In[26]:


#replacing NA with most frequent value which is Feb-2016
data["next_pymnt_d"].fillna('Feb-2016', inplace = True) 


# ##### LETs check the target variable 

# In[27]:


plt.figure(figsize= (12,4))
sns.set_style("whitegrid")
sns.countplot(y='default_ind',data=data)
plt.title("Visualizing Target Variable")
plt.xlabel('COUNT')
plt.ylabel('Fefault_ind')
plt.show()


# In[312]:


data['default_ind'].value_counts()


# Its a highly impalanced target that we have, So for model fitting IMBALANCE treatement would be required here which we will 
# do once we have done the Data cleaning part 

# In[28]:


data.isnull().sum()


# In[29]:


data.columns


# Lets visualize a few parameters after treating Null values

# ##### DATA VISUALIZATION

# In[30]:


dt_series = pd.to_datetime(data['issue_d'])
data['year'] = dt_series.dt.year
data['year'] = data['year'].astype(object)


# In[31]:


# Lets check how much loan amount was issue in which year

plt.figure(figsize= (10,7))
sns.barplot('year', 'loan_amnt', data=data, palette="Blues_d",linewidth=1.5, errcolor=".2", edgecolor=".2")
plt.ylabel('loan amount issued')
plt.xlabel('year of issue')
plt.title('Loan Amount issued v/s Year of iisue')


# Thus it can be concluded that LOAN amount issued kept in increasing with each year.

# In[32]:


# NOW Let's check defaulters yearwise
# Lets check how much loan amount was issue in which year

plt.figure(figsize= (10,7))
sns.barplot('year', 'default_ind', data=data, palette="YlOrRd",linewidth=1.5, errcolor=".2", edgecolor=".2")
plt.ylabel('Defaulters')
plt.xlabel('year of issue')
plt.title('Defaulters v/s Year')


# In[33]:


#grade vs loan amount
plt.figure(figsize= (10,7))
sns.barplot('grade', 'loan_amnt', data=data, palette="GnBu",linewidth=1.5, errcolor=".2", edgecolor=".2")
plt.ylabel('Loan Amount')
plt.xlabel('grade')
plt.title('Loan Amt v/s grade')


# In[34]:


#'home_ownership' vs Loan granted

plt.figure(figsize= (10,7))
sns.barplot('home_ownership','loan_amnt', data=data, palette="Set1",linewidth=1.5, errcolor=".2", edgecolor=".2")
plt.ylabel('Loan Granted')
plt.xlabel('Home Ownershi')
plt.title('Loan Amt v/s grade')


# In[35]:


#purpose vs Loan

plt.figure(figsize= (16,7))
sns.barplot('purpose','loan_amnt', data=data, palette="Dark2",linewidth=1.5, errcolor=".2", edgecolor=".2")
plt.ylabel('Loan Granted')
plt.xlabel('purpose')
plt.title('Loan Amt v/s purpose')


# In[36]:


#  loan given wrt anuual income
plt.plot
plt.figure(figsize= (12,10))
g = sns.regplot(x=data['loan_amnt'], y=data['annual_inc'], fit_reg=False).set_title("LOan amt VS Annual Income")
#it does not convey much


# Above we have visualized a few relations wrt LOAN amount, Now for further EDA We will Diveide data into numerical and 
# categorical and seperately treat them , check their relation wrt target variable and treat possible outliers in them

# In[37]:


data.drop(['year'], axis=1, inplace = True)


# In[38]:


data.columns


# ### Plugging out Numerical Columns and treating them Seperately

# In[39]:


data_num = data.select_dtypes(include = ['float64', 'int64'])


# In[40]:


data_num.shape


# In[41]:


#Using Pearson Correlation
plt.figure(figsize=(22,20))
cor = data_num.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.coolwarm)
plt.show()


# ##### Checking for Multicolinearity & removing the columns with ration greater than 0.8

# In[42]:


corr_matrix = data_num.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
to_drop


# In[43]:


# Drop features 
data_num=data_num.drop(data_num[to_drop], axis=1)


# In[44]:


data_num.drop(['id','policy_code'], axis = 1, inplace = True)
data_num.head()


# In[45]:


data_num.shape


# Checking Correlation again

# In[46]:


#Using Pearson Correlation
plt.figure(figsize=(22,20))
cor = data_num.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.coolwarm)
plt.show()


# #### Now Checking all numerical columns, checking outliers, applying outlier treatement wherever applicable 
# ####  and finally comparing feature wrt Target variable

# In[47]:


#annual_inc
data_num['annual_inc']= data_num['annual_inc'].astype(float)
data_num['annual_inc'].describe()


# In[48]:


#Checking Outlier annual_inc
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['annual_inc'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['annual_inc']).set_title("Before outlier treatement")
plt.show()


# In[49]:


Q1=data_num['annual_inc'].quantile(0.25)
Q3=data_num['annual_inc'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1 - 1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)


# In[50]:


plt.subplots(figsize=(15, 5))
plt.plot
g = sns.distplot(data_num['annual_inc']).set_title("Can't treat outliers here as its important feature")


# In[51]:


print(data_num.shape, data.shape)


# We Observe here that a lot of outliers are there but we cannot apply mean or discard datapoints as its an important indicator so we will kepp all the data points
# 

# In[52]:


data_num['annual_inc_band'] = pd.cut(data_num['annual_inc'], 4)
data_num['annual_inc_band'].value_counts()


# In[53]:


#creating bands
data_num.loc[data_num['annual_inc'] <= 39366.925, 'annual_inc'] = 0
data_num.loc[(data_num['annual_inc'] > 39366.925) & (data_num['annual_inc'] <= 78733.85), 'annual_inc'] = 1
data_num.loc[(data_num['annual_inc'] > 78733.85) & (data_num['annual_inc'] <= 118100.775), 'annual_inc'] = 2
data_num.loc[data_num['annual_inc'] > 118100.775, 'annual_inc'] = 3
data_num['annual_inc'].value_counts()


# In[54]:


sns.countplot(x='annual_inc',hue='default_ind',data=data_num)
plt.tight_layout()


# In[55]:


#loan_amnt
data_num['loan_amnt'].describe()


# In[56]:


#Checking Outlier loan_amnt
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['loan_amnt'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['loan_amnt']).set_title("Bfore outlier treatement")
plt.show()


# In[57]:


Q1=data_num['loan_amnt'].quantile(0.25)
Q3=data_num['loan_amnt'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1 - 1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)


# In[58]:


plt.subplots(figsize=(15, 5))
plt.plot
g = sns.distplot(data_num['loan_amnt']).set_title("NO Outliers exist")

No Outliers exist so we will go ahead with raw Data
# In[59]:


data_num['loan_amnt_band'] = pd.cut(data_num['loan_amnt'], 4)
data_num['loan_amnt_band'].value_counts()


# In[60]:


data_num.loc[data_num['loan_amnt'] <= 9125.0, 'loan_amnt'] = 0
data_num.loc[(data_num['loan_amnt'] > 9125.0) & (data_num['loan_amnt'] <= 17750.0), 'loan_amnt'] = 1
data_num.loc[(data_num['loan_amnt'] > 17750.0) & (data_num['loan_amnt'] <= 26375.0), 'loan_amnt'] = 2
data_num.loc[data_num['loan_amnt'] > 26375.0, 'loan_amnt'] = 3
data_num['loan_amnt'].value_counts()


# In[61]:


sns.countplot(x='loan_amnt',hue='default_ind',data=data_num)
plt.tight_layout()


# In[62]:


#int_rate
data_num['int_rate'].describe()


# In[63]:


#Checking Outlier int_rate
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['int_rate'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['int_rate']).set_title("Before outlier treatement")
plt.show()


# In[64]:


Q1=data_num['int_rate'].quantile(0.25)
Q3=data_num['int_rate'].quantile(0.75)
IQR=Q3-Q1
print(Q1)
print(Q3)
print(IQR)
Lower_Whisker = Q1 - 1.5*IQR
Upper_Whisker = Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)


# In[65]:


plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
g = sns.distplot(data_num['int_rate']).set_title("Before outlier treatement")
data_num = data_num[data_num['int_rate']< Upper_Whisker]
data = data[data['int_rate']< Upper_Whisker]
plt.subplot(1, 2, 2)                                                                                
g = sns.distplot(data_num['int_rate']).set_title("After outlier treatement")


# In[66]:


plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['int_rate'], y=data_num['default_ind'], fit_reg=False).set_title("After outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['int_rate']).set_title("After outlier treatement")
plt.show()


# In[67]:


data_num['int_rate_band'] = pd.cut(data_num['int_rate'], 4)
data_num['int_rate_band'].value_counts()


# In[68]:


data_num.loc[data_num['int_rate'] <= 10.238, 'int_rate'] = 0
data_num.loc[(data_num['int_rate'] > 10.238) & (data_num['int_rate'] <= 15.155), 'int_rate'] = 1
data_num.loc[(data_num['int_rate'] > 15.155) & (data_num['int_rate'] <= 20.072), 'int_rate'] = 2
data_num.loc[data_num['int_rate'] > 20.072, 'int_rate'] = 3
data_num['int_rate'].value_counts()


# In[69]:


sns.countplot(x='int_rate',hue='default_ind',data=data_num)
plt.tight_layout()


# In[70]:


#dti
data_num['dti'].describe()


# In[71]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['dti'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['dti']).set_title("Before outlier treatement")
plt.show()


# In[72]:


# Outliers Treatment
#Find mean of the column "dti"
dti_mean = int(data_num['dti'].mean())
IQR_dti_P75 = data_num['dti'].quantile(q=0.75)
IQR_dti_P25 = data_num['dti'].quantile(q=0.25)
IQR_dti = IQR_dti_P75-IQR_dti_P25
IQR_LL = int(IQR_dti_P25 - 1.5*IQR_dti)
IQR_UL = int(IQR_dti_P75 + 1.5*IQR_dti)


# In[73]:


plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
g = sns.distplot(data_num['dti']).set_title("Before outlier treatement")
data_num.loc[data_num['dti']>IQR_UL , 'dti'] = dti_mean
data.loc[data['dti']>IQR_UL , 'dti'] = dti_mean
data_num.loc[data_num['dti']<IQR_LL , 'dti'] = dti_mean
data.loc[data['dti']<IQR_LL , 'dti'] = dti_mean
plt.subplot(1, 2, 2)                                                                                
g = sns.distplot(data_num['dti']).set_title('After Outlier Treatement')


# In[74]:


plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['dti'], y=data_num['default_ind'], fit_reg=False).set_title("After outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['dti']).set_title("After outlier treatement")
plt.show()


# In[75]:


data_num.shape


# In[76]:


data_num['dti_band'] = pd.cut(data_num['dti'], 4)
data_num['dti_band'].value_counts()


# In[77]:


data_num.loc[data_num['dti'] <= 10.51, 'dti'] = 0
data_num.loc[(data_num['dti'] > 10.51) & (data_num['dti'] <= 21.02), 'dti'] = 1
data_num.loc[(data_num['dti'] > 21.02) & (data_num['dti'] <= 31.53), 'dti'] = 2
data_num.loc[data_num['dti'] > 31.53, 'dti'] = 3
data_num['dti'].value_counts()


# In[78]:


sns.countplot(x='dti',hue='default_ind',data=data_num)
plt.tight_layout()


# In[79]:


#delinq_2yrs
data_num['delinq_2yrs'].describe()


# In[80]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['delinq_2yrs'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['delinq_2yrs']).set_title("Before outlier treatement")
plt.show()


# In[81]:


print(data.shape, data_num.shape)


# #### Outliers Treatment not required here as its imp para wrt corruption

# In[82]:


plt.subplots(figsize=(15, 5))
plt.plot
g = sns.distplot(data_num['delinq_2yrs']).set_title("outlier treatement not required here")


# Conclusion : Outlier treatement should not be required here as all data point are required here

# In[83]:


data_num['delinq_2yrs_bandvx'] = pd.cut(data_num['delinq_2yrs'], 4)
data_num['delinq_2yrs_bandvx'].value_counts()


# In[84]:


data_num.loc[data_num['delinq_2yrs'] <= 0, 'delinq_2yrs'] = 0
data_num.loc[data_num['delinq_2yrs'] > 0, 'delinq_2yrs'] = 1
data_num['delinq_2yrs'].value_counts()


# In[85]:


sns.countplot(x='delinq_2yrs',hue='default_ind',data=data_num)
plt.tight_layout()  


# In[86]:


#inq_last_6mths
data_num['inq_last_6mths'].describe()


# In[87]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['inq_last_6mths'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['inq_last_6mths']).set_title("Before outlier treatement")
plt.show()


# In[88]:


# Outliers Treatment
#Find mean of the column "inq_last_6mths"
inq_last_6mths_mean = int(data_num['inq_last_6mths'].mean())

#FInd 75th Percentile of the column "inq_last_6mths"
IQR_inq_last_6mths_P75 = data_num['inq_last_6mths'].quantile(q=0.75)

#FInd 25th Percentile of the column "inq_last_6mths"
IQR_inq_last_6mths_P25 = data_num['inq_last_6mths'].quantile(q=0.25)

#FInd IQR of the column "inq_last_6mths"
IQR_inq_last_6mths = IQR_inq_last_6mths_P75-IQR_inq_last_6mths_P25

#Fix boundaries to detect outliers in column "dti"
IQR_LL = int(IQR_inq_last_6mths_P25 - 1.5*IQR_inq_last_6mths)
IQR_UL = int(IQR_inq_last_6mths_P75 + 1.5*IQR_inq_last_6mths)


# In[89]:


plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
g = sns.distplot(data_num['inq_last_6mths']).set_title("Before outlier treatement")

#treating upper end outier with mean
data_num.loc[data_num['inq_last_6mths']>IQR_UL , 'inq_last_6mths'] = inq_last_6mths_mean
data.loc[data['inq_last_6mths']>IQR_UL , 'inq_last_6mths'] = inq_last_6mths_mean

#treating lower end outlier as mean
data_num.loc[data_num['inq_last_6mths']<IQR_LL , 'inq_last_6mths'] = inq_last_6mths_mean
data.loc[data['inq_last_6mths']<IQR_LL , 'inq_last_6mths'] = inq_last_6mths_mean
plt.subplot(1, 2, 2)                                                                                
g = sns.distplot(data_num['inq_last_6mths']).set_title("After outlier treatement")


# In[90]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['inq_last_6mths'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['inq_last_6mths']).set_title("Before outlier treatement")
plt.show()


# In[91]:


#inq_last_6mths
data_num['inq_last_6mths_band'] = pd.cut(data_num['inq_last_6mths'], 4)
data_num['inq_last_6mths_band'].value_counts()


# In[92]:


data_num.loc[data_num['inq_last_6mths'] <= 0.5, 'inq_last_6mths'] = 0
data_num.loc[(data_num['inq_last_6mths'] > 0.5) & (data_num['inq_last_6mths'] <= 1.0), 'inq_last_6mths'] = 1
data_num.loc[data_num['inq_last_6mths'] > 1.0, 'inq_last_6mths'] = 2
data_num['inq_last_6mths'].value_counts()


# In[93]:


sns.countplot(x='inq_last_6mths',hue='default_ind',data=data_num)
plt.tight_layout()


# In[94]:


#mths_since_last_delinq
data_num['mths_since_last_delinq'].describe()


# In[95]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['mths_since_last_delinq'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['mths_since_last_delinq']).set_title("Before outlier treatement")
plt.show()


# In[96]:


# Outliers Treatment
#Find mean of the column "mths_since_last_delinq"
mths_since_last_delinq_mean = int(data_num['mths_since_last_delinq'].mean())

#FInd 75th Percentile of the column "mths_since_last_delinq"
IQR_mths_since_last_delinq_P75 = data_num['mths_since_last_delinq'].quantile(q=0.75)

#FInd 25th Percentile of the column "mths_since_last_delinq"
IQR_mths_since_last_delinq_P25 = data_num['mths_since_last_delinq'].quantile(q=0.25)

#FInd IQR of the column "mths_since_last_delinq"
IQR_mths_since_last_delinq = IQR_mths_since_last_delinq_P75-IQR_mths_since_last_delinq_P25

#Fix boundaries to detect outliers in column "dti"
IQR_LL = int(IQR_mths_since_last_delinq_P25 - 1.5*IQR_mths_since_last_delinq)
IQR_UL = int(IQR_mths_since_last_delinq_P75 + 1.5*IQR_mths_since_last_delinq)


# In[97]:


plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
g = sns.distplot(data_num['mths_since_last_delinq']).set_title("Before outlier treatement")

#treating upper end outier with mean
data_num.loc[data_num['mths_since_last_delinq']>IQR_UL , 'mths_since_last_delinq'] = mths_since_last_delinq_mean
data.loc[data['mths_since_last_delinq']>IQR_UL , 'mths_since_last_delinq'] = mths_since_last_delinq_mean

#treating lower end outlier as mean
data_num.loc[data_num['mths_since_last_delinq']<IQR_LL , 'mths_since_last_delinq'] = mths_since_last_delinq_mean
data.loc[data['mths_since_last_delinq']<IQR_LL , 'mths_since_last_delinq'] = mths_since_last_delinq_mean
plt.subplot(1, 2, 2)                                                                                
g = sns.distplot(data_num['mths_since_last_delinq']).set_title("After outlier treatement")


# In[98]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['mths_since_last_delinq'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['mths_since_last_delinq']).set_title("Before outlier treatement")
plt.show()


# In[99]:


#mths_since_last_delinq
data_num['mths_since_last_delinq_band'] = pd.cut(data_num['mths_since_last_delinq'], 4)
data_num['mths_since_last_delinq_band'].value_counts()


# In[100]:


data_num.loc[data_num['mths_since_last_delinq'] <= 31.25, 'mths_since_last_delinq'] = 0
data_num.loc[data_num['mths_since_last_delinq'] > 31.25, 'mths_since_last_delinq'] = 1
data_num['mths_since_last_delinq'].value_counts()


# In[101]:


sns.countplot(x='mths_since_last_delinq',hue='default_ind',data=data_num)
plt.tight_layout()


# In[102]:


#open_acc
data_num['open_acc'].describe()


# In[103]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['open_acc'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['open_acc']).set_title("Before outlier treatement")
plt.show()


# In[104]:


# Outliers Treatment
#Find mean of the column "open_acc"
open_acc_mean = int(data_num['open_acc'].mean())

#FInd 75th Percentile of the column "open_acc"
IQR_open_acc_P75 = data_num['open_acc'].quantile(q=0.75)

#FInd 25th Percentile of the column "open_acc"
IQR_open_acc_P25 = data_num['open_acc'].quantile(q=0.25)

#FInd IQR of the column "open_acc"
IQR_open_acc = IQR_open_acc_P75-IQR_open_acc_P25

#Fix boundaries to detect outliers in column "dti"
IQR_LL = int(IQR_open_acc_P25 - 1.5*IQR_open_acc)
IQR_UL = int(IQR_open_acc_P75 + 1.5*IQR_open_acc)


# In[105]:


plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
g = sns.distplot(data_num['open_acc']).set_title("Before outlier treatement")

#treating upper end outier with mean
data_num.loc[data_num['open_acc']>IQR_UL , 'open_acc'] = open_acc_mean
data.loc[data['open_acc']>IQR_UL , 'open_acc'] = open_acc_mean

#treating lower end outlier as mean
data_num.loc[data_num['open_acc']<IQR_LL , 'open_acc'] = open_acc_mean
data.loc[data['open_acc']<IQR_LL , 'open_acc'] = open_acc_mean
plt.subplot(1, 2, 2)                                                                                
g = sns.distplot(data_num['open_acc']).set_title("After outlier treatement")


# In[106]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['open_acc'], y=data_num['default_ind'], fit_reg=False).set_title("After outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['open_acc']).set_title("After outlier treatement")
plt.show()


# In[107]:


#open_acc
data_num['open_acc_ba'] = pd.cut(data_num['open_acc'], 4)
data_num['open_acc_ba'].value_counts()


# In[108]:


data_num.loc[data_num['open_acc'] <= 5.75, 'open_acc'] = 0
data_num.loc[(data_num['open_acc'] > 5.75) & (data_num['open_acc'] <= 11.5), 'open_acc'] = 1
data_num.loc[(data_num['open_acc'] > 11.5) & (data_num['open_acc'] <= 17.25), 'open_acc'] = 2
data_num.loc[data_num['open_acc'] > 17.25, 'open_acc'] = 3
data_num['open_acc'].value_counts()


# In[109]:


sns.countplot(x='open_acc',hue='default_ind',data=data_num)
plt.tight_layout()


# In[110]:


#pub_rec
data_num['pub_rec'].describe()


# In[111]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['pub_rec'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['pub_rec']).set_title("Before outlier treatement")
plt.show()


# ###### Outliers Treatment - have to binarize so cant treat outlier

# In[112]:


plt.subplots(figsize=(15, 5))
plt.plot
g = sns.distplot(data_num['pub_rec']).set_title("Outlier treatement not req")


# In[113]:


data_num['pub_rec_ban'] = pd.cut(data_num['pub_rec'], 4)
data_num['pub_rec_ban'].value_counts()


# In[114]:


data_num.loc[data_num['pub_rec'] <= 0, 'pub_rec'] = 0
data_num.loc[data_num['pub_rec'] > 0, 'pub_rec'] = 1
data_num['open_acc'].value_counts()


# In[115]:


sns.countplot(x='pub_rec',hue='default_ind',data=data_num)
plt.tight_layout() #irrelevant can be removed if we want


# In[116]:


#revol_bal
data_num['revol_bal'].describe()


# In[117]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['revol_bal'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['revol_bal']).set_title("Before outlier treatement")
plt.show()


# In[118]:


# Outliers Treatment
#Find mean of the column "revol_bal"
open_revol_bal = int(data_num['revol_bal'].mean())

#FInd 75th Percentile of the column "revol_bal"
IQR_revol_bal_P75 = data_num['revol_bal'].quantile(q=0.75)

#FInd 25th Percentile of the column "revol_bal"
IQR_revol_bal_P25 = data_num['revol_bal'].quantile(q=0.25)

#FInd IQR of the column "revol_bal"
IQR_revol_bal = IQR_revol_bal_P75-IQR_revol_bal_P25

#Fix boundaries to detect outliers in column "dti"
IQR_LL = int(IQR_revol_bal_P25 - 1.5*IQR_revol_bal)
IQR_UL = int(IQR_revol_bal_P75 + 1.5*IQR_revol_bal)


# In[119]:


plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
g = sns.distplot(data_num['revol_bal']).set_title("After outlier treatement")

#treating upper end outier with mean
data_num.loc[data_num['revol_bal']>IQR_UL , 'revol_bal'] = open_revol_bal
data.loc[data['revol_bal']>IQR_UL , 'revol_bal'] = open_revol_bal

#treating lower end outlier as mean
data_num.loc[data_num['revol_bal']<IQR_LL , 'revol_bal'] = open_revol_bal
data.loc[data['revol_bal']<IQR_LL , 'revol_bal'] = open_revol_bal
plt.subplot(1, 2, 2)                                                                                
g = sns.distplot(data_num['revol_bal']).set_title("After outlier treatement")


# In[120]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['revol_bal'], y=data_num['default_ind'], fit_reg=False).set_title("After outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['revol_bal']).set_title("After outlier treatement")
plt.show()


# In[121]:


data_num['revol_bal_baad'] = pd.cut(data_num['revol_bal'], 4)
data_num['revol_bal_baad'].value_counts()


# In[122]:


data_num.loc[data_num['revol_bal'] <= 10118.25, 'revol_bal'] = 0
data_num.loc[(data_num['revol_bal'] > 10118.25) & (data_num['revol_bal'] <= 20236.5), 'revol_bal'] = 1
data_num.loc[(data_num['revol_bal'] > 20236.5) & (data_num['revol_bal'] <= 30354.75), 'revol_bal'] = 2
data_num.loc[data_num['revol_bal'] > 17.25, 'revol_bal'] = 3
data_num['revol_bal'].value_counts()


# In[123]:


sns.countplot(x='revol_bal',hue='default_ind',data=data_num)
plt.tight_layout()


# In[124]:


#revol_util
data_num['revol_util'].describe()


# In[125]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['revol_util'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['revol_util']).set_title("Before outlier treatement")
plt.show()


# In[126]:


# Outliers Treatment
#Find mean of the column "revol_util"
revol_util_mean = int(data_num['revol_util'].mean())

#FInd 75th Percentile of the column "revol_util"
IQR_revol_util_P75 = data_num['revol_util'].quantile(q=0.75)

#FInd 25th Percentile of the column "revol_util"
IQR_revol_util_P25 = data_num['revol_util'].quantile(q=0.25)

#FInd IQR of the column "open_acc"
IQR_revol_util = IQR_revol_util_P75-IQR_revol_util_P25

#Fix boundaries to detect outliers in column "dti"
IQR_LL = int(IQR_revol_util_P25 - 1.5*IQR_revol_util)
IQR_UL = int(IQR_revol_util_P75 + 1.5*IQR_revol_util)


# In[127]:


plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
g = sns.distplot(data_num['revol_util']).set_title("After outlier treatement")

#treating upper end outier with mean
data_num.loc[data_num['revol_util']>IQR_UL , 'revol_util'] = revol_util_mean
data.loc[data['revol_util']>IQR_UL , 'revol_util'] = revol_util_mean

#treating lower end outlier as mean
data_num.loc[data_num['revol_util']<IQR_LL , 'revol_util'] = revol_util_mean
data.loc[data['revol_util']<IQR_LL , 'revol_util'] = revol_util_mean
plt.subplot(1, 2, 2)                                                                                
g = sns.distplot(data_num['revol_util']).set_title("After outlier treatement")


# In[128]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['revol_util'], y=data_num['default_ind'], fit_reg=False).set_title("After outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['revol_util']).set_title("After outlier treatement")
plt.show()


# In[129]:


data_num['revol_util_ban'] = pd.cut(data_num['revol_util'], 3)
data_num['revol_util_ban'].value_counts()


# In[130]:


data_num.loc[data_num['revol_util'] <= 42.0, 'revol_util'] = 0
data_num.loc[(data_num['revol_util'] > 42.0) & (data_num['revol_util'] <= 84.0), 'revol_util'] = 1
data_num.loc[data_num['revol_util'] > 84.0, 'revol_util'] = 2
data_num['revol_util'].value_counts()


# In[131]:


sns.countplot(x='revol_util',hue='default_ind',data=data_num)
plt.tight_layout()


# In[132]:


#total_acc
data_num['total_acc'].describe()


# In[133]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['total_acc'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['total_acc']).set_title("Before outlier treatement")
plt.show()


# In[134]:


# Outliers Treatment
#Find mean of the column "total_acc"
total_acc_mean = int(data_num['total_acc'].mean())

#FInd 75th Percentile of the column "total_acc"
IQR_total_acc_P75 = data_num['total_acc'].quantile(q=0.75)

#FInd 25th Percentile of the column "total_acc"
IQR_total_acc_P25 = data_num['total_acc'].quantile(q=0.25)

#FInd IQR of the column "total_acc"
IQR_total_acc = IQR_total_acc_P75-IQR_total_acc_P25

#Fix boundaries to detect outliers in column "dti"
IQR_LL = int(IQR_total_acc_P25 - 1.5*IQR_total_acc)
IQR_UL = int(IQR_total_acc_P75 + 1.5*IQR_total_acc)


# In[135]:


plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
g = sns.distplot(data_num['total_acc']).set_title("After outlier treatement")

#treating upper end outier with mean
data_num.loc[data_num['total_acc']>IQR_UL , 'total_acc'] = total_acc_mean
data.loc[data['total_acc']>IQR_UL , 'total_acc'] = total_acc_mean

#treating lower end outlier as mean
data_num.loc[data_num['total_acc']<IQR_LL , 'total_acc'] = total_acc_mean
data.loc[data['total_acc']<IQR_LL , 'total_acc'] = total_acc_mean
plt.subplot(1, 2, 2)                                                                                
g = sns.distplot(data_num['total_acc']).set_title("After outlier treatement")


# In[136]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['total_acc'], y=data_num['default_ind'], fit_reg=False).set_title("After outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['total_acc']).set_title("After outlier treatement")
plt.show()


# In[137]:


data_num['total_acc_band'] = pd.cut(data_num['total_acc'], 3)
data_num['total_acc_band'].value_counts()


# In[138]:


data_num.loc[data_num['total_acc'] <= 20.0, 'total_acc'] = 0
data_num.loc[(data_num['total_acc'] > 20.0) & (data_num['total_acc'] <= 38.0), 'total_acc'] = 1
data_num.loc[data_num['total_acc'] > 38.0, 'total_acc'] = 2
data_num['total_acc'].value_counts()


# In[139]:


sns.countplot(x='total_acc',hue='default_ind',data=data_num)
plt.tight_layout()


# In[140]:


#out_prncp
data_num['out_prncp'].describe()


# In[141]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['out_prncp'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['out_prncp']).set_title("Before outlier treatement")
plt.show()


# In[142]:


# Outliers Treatment
#Find mean of the column "out_prncp"
out_prncp_mean = int(data_num['out_prncp'].mean())

#FInd 75th Percentile of the column "out_prncp"
IQR_out_prncp_P75 = data_num['out_prncp'].quantile(q=0.75)

#FInd 25th Percentile of the column "out_prncp"
IQR_out_prncp_P25 = data_num['out_prncp'].quantile(q=0.25)

#FInd IQR of the column "open_acc"
IQR_out_prncp = IQR_out_prncp_P75-IQR_out_prncp_P25

#Fix boundaries to detect outliers in column "dti"
IQR_LL = int(IQR_out_prncp_P25 - 1.5*IQR_out_prncp)
IQR_UL = int(IQR_out_prncp_P75 + 1.5*IQR_out_prncp)


# In[143]:


plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
g = sns.distplot(data_num['out_prncp']).set_title("After outlier treatement")

#treating upper end outier with mean
data_num.loc[data_num['out_prncp']>IQR_UL , 'out_prncp'] = out_prncp_mean
data.loc[data['out_prncp']>IQR_UL , 'out_prncp'] = out_prncp_mean

#treating lower end outlier as mean
data_num.loc[data_num['out_prncp']<IQR_LL , 'out_prncp'] = out_prncp_mean
data.loc[data['out_prncp']<IQR_LL , 'out_prncp'] = out_prncp_mean
plt.subplot(1, 2, 2)                                                                                
g = sns.distplot(data_num['out_prncp']).set_title("After outlier treatement")


# In[144]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['out_prncp'], y=data_num['default_ind'], fit_reg=False).set_title("After outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['out_prncp']).set_title("After outlier treatement")
plt.show()


# In[145]:


data_num['out_prncp_ban'] = pd.cut(data_num['out_prncp'], 3)
data_num['out_prncp_ban'].value_counts()


# In[146]:


data_num.loc[data_num['out_prncp'] <= 10854.23, 'out_prncp'] = 0
data_num.loc[(data_num['out_prncp'] > 10854.23) & (data_num['out_prncp'] <= 21708.46), 'out_prncp'] = 1
data_num.loc[data_num['out_prncp'] > 21708.46, 'out_prncp'] = 2
data_num['out_prncp'].value_counts()


# In[147]:


sns.countplot(x='out_prncp',hue='default_ind',data=data_num)
plt.tight_layout()


# In[148]:


#total_pymnt
data_num['total_pymnt'].describe()


# In[149]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['total_pymnt'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['total_pymnt']).set_title("Before outlier treatement")
plt.show()


# In[150]:


# Outliers Treatment
#Find mean of the column "total_pymnt"
total_pymnt_mean = int(data_num['total_pymnt'].mean())

#FInd 75th Percentile of the column "total_pymnt"
IQR_total_pymnt_P75 = data_num['total_pymnt'].quantile(q=0.75)

#FInd 25th Percentile of the column "total_pymnt"
IQR_total_pymnt_P25 = data_num['total_pymnt'].quantile(q=0.25)

#FInd IQR of the column "total_pymnt"
IQR_total_pymnt = IQR_total_pymnt_P75-IQR_total_pymnt_P25

#Fix boundaries to detect outliers in column "dti"
IQR_LL = int(IQR_total_pymnt_P25 - 1.5*IQR_total_pymnt)
IQR_UL = int(IQR_total_pymnt_P75 + 1.5*IQR_total_pymnt)


# In[151]:


plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
g = sns.distplot(data_num['total_pymnt']).set_title("After outlier treatement")

#treating upper end outier with mean
data_num.loc[data_num['total_pymnt']>IQR_UL , 'total_pymnt'] = total_pymnt_mean
data.loc[data['total_pymnt']>IQR_UL , 'total_pymnt'] = total_pymnt_mean

#treating lower end outlier as mean
data_num.loc[data_num['total_pymnt']<IQR_LL , 'total_pymnt'] = total_pymnt_mean
data.loc[data['total_pymnt']<IQR_LL , 'total_pymnt'] = total_pymnt_mean
plt.subplot(1, 2, 2)                                                                                
g = sns.distplot(data_num['total_pymnt']).set_title("After outlier treatement")


# In[152]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['total_pymnt'], y=data_num['default_ind'], fit_reg=False).set_title("After outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['total_pymnt']).set_title("After outlier treatement")
plt.show()


# In[153]:


data_num['total_pymnt_ban'] = pd.cut(data_num['total_pymnt'], 3)
data_num['total_pymnt_ban'].value_counts()


# In[154]:


data_num.loc[data_num['total_pymnt'] <= 7796.647, 'total_pymnt'] = 0
data_num.loc[(data_num['total_pymnt'] > 7796.647) & (data_num['total_pymnt'] <= 15593.293), 'total_pymnt'] = 1
data_num.loc[data_num['total_pymnt'] > 15593.293, 'total_pymnt'] = 2
data_num['total_pymnt'].value_counts()


# In[155]:


sns.countplot(x='total_pymnt',hue='default_ind',data=data_num)
plt.tight_layout()


# In[156]:


#total_rec_int
data_num['total_rec_int'].describe()


# In[157]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['total_rec_int'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['total_rec_int']).set_title("Before outlier treatement")
plt.show()


# In[158]:


# Outliers Treatment
#Find mean of the column "total_rec_int"
total_rec_int_mean = int(data_num['total_rec_int'].mean())

#FInd 75th Percentile of the column "total_rec_int"
IQR_total_rec_int_P75 = data_num['total_rec_int'].quantile(q=0.75)

#FInd 25th Percentile of the column "total_rec_int"
IQR_total_rec_int_P25 = data_num['total_rec_int'].quantile(q=0.25)

#FInd IQR of the column "total_rec_int"
IQR_total_rec_int = IQR_total_rec_int_P75-IQR_total_rec_int_P25

#Fix boundaries to detect outliers in column "dti"
IQR_LL = int(IQR_total_rec_int_P25 - 1.5*IQR_total_rec_int)
IQR_UL = int(IQR_total_rec_int_P75 + 1.5*IQR_total_rec_int)


# In[159]:


plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
g = sns.distplot(data_num['total_rec_int']).set_title("After outlier treatement")

#treating upper end outier with mean
data_num.loc[data_num['total_rec_int']>IQR_UL , 'total_rec_int'] = total_rec_int_mean
data.loc[data['total_rec_int']>IQR_UL , 'total_rec_int'] = total_rec_int_mean

#treating lower end outlier as mean
data_num.loc[data_num['total_rec_int']<IQR_LL , 'total_rec_int'] = total_rec_int_mean
data.loc[data['total_rec_int']<IQR_LL , 'total_rec_int'] = total_rec_int_mean
plt.subplot(1, 2, 2)                                                                                
g = sns.distplot(data_num['total_rec_int']).set_title("After outlier treatement")


# In[160]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['total_rec_int'], y=data_num['default_ind'], fit_reg=False).set_title("After outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['total_rec_int']).set_title("After outlier treatement")
plt.show()


# In[161]:


data_num['total_rec_int_band'] = pd.cut(data_num['total_rec_int'], 4)
data_num['total_rec_int_band'].value_counts()


# In[162]:


data_num.loc[data_num['total_rec_int'] <= 1190.965, 'total_rec_int'] = 0
data_num.loc[(data_num['total_rec_int'] > 1190.965) & (data_num['total_rec_int'] <= 2381.93), 'total_rec_int'] = 1
data_num.loc[data_num['total_rec_int'] > 2381.93, 'total_rec_int'] = 2
data_num['total_rec_int'].value_counts()


# In[163]:


sns.countplot(x='total_rec_int',hue='default_ind',data=data_num)
plt.tight_layout()


# In[164]:


#total_rec_late_fee
data_num['total_rec_late_fee'].describe()


# In[165]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['total_rec_late_fee'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['total_rec_late_fee']).set_title("Before outlier treatement")
plt.show()


# ##### Outliers Treatment not required

# In[166]:


plt.subplots(figsize=(15, 5))
plt.plot
g = sns.distplot(data_num['total_rec_late_fee']).set_title("outlier treatement not required")


# In[167]:


data_num['total_rec_late_fee_band'] = pd.cut(data_num['total_rec_late_fee'], 4)
data_num['total_rec_late_fee_band'].value_counts()# Should be removed as it doesnot convey anything


# In[168]:


data_num.loc[data_num['total_rec_late_fee'] <= 0, 'total_rec_late_fee'] = 0
data_num.loc[data_num['total_rec_late_fee'] > 0, 'total_rec_late_fee'] = 1
data_num['total_rec_late_fee'].value_counts()


# In[169]:


sns.countplot(x='total_rec_late_fee',hue='default_ind',data=data_num)
plt.tight_layout() 


# In[170]:


#recoveries
data_num['recoveries'].describe()


# In[171]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['recoveries'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['recoveries']).set_title("Before outlier treatement")
plt.show()


# In[172]:


# Outliers Treatment
#Find mean of the column "recoveries"
recoveries_mean = int(data_num['recoveries'].mean())

#FInd 75th Percentile of the column "recoveries"
IQR_recoveries_P75 = data_num['recoveries'].quantile(q=0.75)

#FInd 25th Percentile of the column "recoveries"
IQR_recoveries_P25 = data_num['recoveries'].quantile(q=0.25)

#FInd IQR of the column "recoveries"
IQR_recoveries = IQR_recoveries_P75-IQR_recoveries_P25

#Fix boundaries to detect outliers in column "dti"
IQR_LL = int(IQR_recoveries_P25 - 1.5*IQR_recoveries)
IQR_UL = int(IQR_recoveries_P75 + 1.5*IQR_recoveries)


# In[173]:


plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
g = sns.distplot(data_num['recoveries']).set_title("After outlier treatement")

#treating upper end outier with mean
data_num.loc[data_num['recoveries']>IQR_UL , 'recoveries'] = recoveries_mean
data.loc[data['recoveries']>IQR_UL , 'recoveries'] = recoveries_mean

#treating lower end outlier as mean
data_num.loc[data_num['recoveries']<IQR_LL , 'recoveries'] = recoveries_mean
data.loc[data['recoveries']<IQR_LL , 'recoveries'] = recoveries_mean
plt.subplot(1, 2, 2)                                                                                
g = sns.distplot(data_num['recoveries']).set_title("After outlier treatement")


# In[174]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['recoveries'], y=data_num['default_ind'], fit_reg=False).set_title("After outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['recoveries']).set_title("After outlier treatement")
plt.show()


# In[175]:


data_num['recoveries_band'] = pd.cut(data_num['recoveries'], 4)
data_num['recoveries_band'].value_counts()


# In[176]:


data_num.loc[data_num['recoveries'] <= 11.25, 'recoveries'] = 0
data_num.loc[data_num['recoveries'] > 11.25, 'recoveries'] = 1
data_num['recoveries'].value_counts()


# In[177]:


sns.countplot(x='recoveries',hue='default_ind',data=data_num)
plt.tight_layout()


# In[178]:


#last_pymnt_amnt
data_num['last_pymnt_amnt'].describe()


# In[179]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['last_pymnt_amnt'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['last_pymnt_amnt']).set_title("Before outlier treatement")
plt.show()


# In[180]:


# Outliers Treatment
#Find mean of the column "last_pymnt_amnt"
last_pymnt_amnt_mean = int(data_num['last_pymnt_amnt'].mean())

#FInd 75th Percentile of the column "last_pymnt_amnt"
IQR_last_pymnt_amnt_P75 = data_num['last_pymnt_amnt'].quantile(q=0.75)

#FInd 25th Percentile of the column "last_pymnt_amnt"
IQR_last_pymnt_amnt_P25 = data_num['last_pymnt_amnt'].quantile(q=0.25)

#FInd IQR of the column "last_pymnt_amnt"
IQR_last_pymnt_amnt = IQR_last_pymnt_amnt_P75-IQR_last_pymnt_amnt_P25

#Fix boundaries to detect outliers in column "dti"
IQR_LL = int(IQR_last_pymnt_amnt_P25 - 1.5*IQR_last_pymnt_amnt)
IQR_UL = int(IQR_last_pymnt_amnt_P75 + 1.5*IQR_last_pymnt_amnt)


# In[181]:


plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
g = sns.distplot(data_num['last_pymnt_amnt']).set_title("After outlier treatement")

#treating upper end outier with mean
data_num.loc[data_num['last_pymnt_amnt']>IQR_UL , 'last_pymnt_amnt'] = last_pymnt_amnt_mean
data.loc[data['last_pymnt_amnt']>IQR_UL , 'last_pymnt_amnt'] = last_pymnt_amnt_mean

#treating lower end outlier as mean
data_num.loc[data_num['last_pymnt_amnt']<IQR_LL , 'last_pymnt_amnt'] = last_pymnt_amnt_mean
data.loc[data['last_pymnt_amnt']<IQR_LL , 'last_pymnt_amnt'] = last_pymnt_amnt_mean
plt.subplot(1, 2, 2)                                                                                
g = sns.distplot(data_num['last_pymnt_amnt']).set_title("After outlier treatement")


# In[182]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['last_pymnt_amnt'], y=data_num['default_ind'], fit_reg=False).set_title("After outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['last_pymnt_amnt']).set_title("After outlier treatement")
plt.show()


# In[183]:


data_num['last_pymnt_amnt_ban'] = pd.cut(data_num['last_pymnt_amnt'], 3)
data_num['last_pymnt_amnt_ban'].value_counts()


# In[184]:


data_num.loc[data_num['last_pymnt_amnt'] <= 359.33, 'last_pymnt_amnt'] = 0
data_num.loc[data_num['last_pymnt_amnt'] > 359.33, 'last_pymnt_amnt'] = 1
data_num['last_pymnt_amnt'].value_counts()


# In[185]:


sns.countplot(x='last_pymnt_amnt',hue='default_ind',data=data_num)
plt.tight_layout()


# In[186]:


#collections_12_mths_ex_med
data_num['collections_12_mths_ex_med'].describe()


# In[187]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['collections_12_mths_ex_med'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['collections_12_mths_ex_med']).set_title("Before outlier treatement")
plt.show()


# ##### Outliers Treatment- not required here because of nature of column

# In[188]:


plt.subplots(figsize=(15, 5))
plt.plot
g = sns.distplot(data_num['collections_12_mths_ex_med']).set_title("Uutlier treatement not required")


# In[189]:


data_num['collections_12_mths_ex_med_band'] = pd.cut(data_num['collections_12_mths_ex_med'], 4)
data_num['collections_12_mths_ex_med_band'].value_counts()


# In[190]:


data_num.loc[data_num['collections_12_mths_ex_med'] <= 0, 'collections_12_mths_ex_med'] = 0
data_num.loc[data_num['collections_12_mths_ex_med'] > 0, 'collections_12_mths_ex_med'] = 1
data_num['collections_12_mths_ex_med'].value_counts()


# In[191]:


sns.countplot(x='collections_12_mths_ex_med',hue='default_ind',data=data_num)
plt.tight_layout() # can be dropped does not infer much


# In[192]:


#acc_now_delinq
data_num['acc_now_delinq'].describe()


# In[193]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['acc_now_delinq'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['acc_now_delinq']).set_title("Before outlier treatement")
plt.show()


# ##### Outliers Treatment not req as o/p needs to be in binarize format

# In[194]:


plt.subplots(figsize=(15, 5))
plt.plot
g = sns.distplot(data_num['acc_now_delinq']).set_title("Outlier treatement not req")


# In[195]:


data_num.loc[data_num['acc_now_delinq'] <= 0, 'acc_now_delinq'] = 0
data_num.loc[data_num['acc_now_delinq'] > 0, 'acc_now_delinq'] = 1
data_num['acc_now_delinq'].value_counts()


# In[196]:


sns.countplot(x='acc_now_delinq',hue='default_ind',data=data_num)
plt.tight_layout() #can be dropped as it doesnot convey anything


# In[197]:


#tot_coll_amt
data_num['tot_coll_amt'].describe()


# In[198]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['tot_coll_amt'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['tot_coll_amt']).set_title("Before outlier treatement")
plt.show()


# In[199]:


# Outliers Treatment
#Find mean of the column "tot_coll_amt"
tot_coll_amt_mean = int(data_num['tot_coll_amt'].mean())

#FInd 75th Percentile of the column "tot_coll_amt"
IQR_tot_coll_amt_P75 = data_num['tot_coll_amt'].quantile(q=0.75)

#FInd 25th Percentile of the column "tot_coll_amt"
IQR_tot_coll_amt_P25 = data_num['tot_coll_amt'].quantile(q=0.25)

#FInd IQR of the column "tot_coll_amt"
IQR_tot_coll_amt = IQR_tot_coll_amt_P75-IQR_tot_coll_amt_P25

#Fix boundaries to detect outliers in column "tot_coll_amt"
IQR_LL = int(IQR_tot_coll_amt_P25 - 1.5*IQR_tot_coll_amt)
IQR_UL = int(IQR_tot_coll_amt_P75 + 1.5*IQR_tot_coll_amt)


# In[200]:


plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
g = sns.distplot(data_num['tot_coll_amt']).set_title("After outlier treatement")

#treating upper end outier with mean
data_num.loc[data_num['tot_coll_amt']>IQR_UL , 'tot_coll_amt'] = tot_coll_amt_mean
data.loc[data['tot_coll_amt']>IQR_UL , 'tot_coll_amt'] = tot_coll_amt_mean

#treating lower end outlier as mean
data_num.loc[data_num['tot_coll_amt']<IQR_LL , 'tot_coll_amt'] = tot_coll_amt_mean
data.loc[data['tot_coll_amt']<IQR_LL , 'tot_coll_amt'] = tot_coll_amt_mean
plt.subplot(1, 2, 2)                                                                                
g = sns.distplot(data_num['tot_coll_amt']).set_title("After outlier treatement")


# In[201]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['tot_coll_amt'], y=data_num['default_ind'], fit_reg=False).set_title("After outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['tot_coll_amt']).set_title("After outlier treatement")
plt.show()


# In[202]:


data_num['tot_coll_amt_band'] = pd.cut(data_num['tot_coll_amt'], 4)
data_num['tot_coll_amt_band'].value_counts()


# In[203]:


data_num.loc[data_num['tot_coll_amt'] <= 56.25, 'tot_coll_amt'] = 0
data_num.loc[data_num['tot_coll_amt'] > 56.25, 'tot_coll_amt'] = 1
data_num['tot_coll_amt'].value_counts()


# In[204]:


sns.countplot(x='tot_coll_amt',hue='default_ind',data=data_num)
plt.tight_layout()


# In[205]:


#tot_cur_bal
data_num['tot_cur_bal'].describe()


# In[206]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['tot_cur_bal'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['tot_cur_bal']).set_title("Before outlier treatement")
plt.show()


# In[207]:


# Outliers Treatment
#Find mean of the column "tot_cur_bal"
tot_cur_bal_mean = int(data_num['tot_cur_bal'].mean())

#FInd 75th Percentile of the column "tot_cur_bal"
IQR_tot_cur_bal_P75 = data_num['tot_cur_bal'].quantile(q=0.75)

#FInd 25th Percentile of the column "tot_cur_bal"
IQR_tot_cur_bal_P25 = data_num['tot_cur_bal'].quantile(q=0.25)

#FInd IQR of the column "tot_cur_bal"
IQR_tot_cur_bal = IQR_tot_cur_bal_P75-IQR_tot_cur_bal_P25

#Fix boundaries to detect outliers in column "tot_cur_bal"
IQR_LL = int(IQR_tot_cur_bal_P25 - 1.5*IQR_tot_cur_bal)
IQR_UL = int(IQR_tot_cur_bal_P75 + 1.5*IQR_tot_cur_bal)


# In[208]:


plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
g = sns.distplot(data_num['tot_cur_bal']).set_title("After outlier treatement")

#treating upper end outier with mean
data_num.loc[data_num['tot_cur_bal']>IQR_UL , 'tot_cur_bal'] = tot_cur_bal_mean
data.loc[data['tot_cur_bal']>IQR_UL , 'tot_cur_bal'] = tot_cur_bal_mean

#treating lower end outlier as mean
data_num.loc[data_num['tot_cur_bal']<IQR_LL , 'tot_cur_bal'] = tot_cur_bal_mean
data.loc[data['tot_cur_bal']<IQR_LL , 'tot_cur_bal'] = tot_cur_bal_mean
plt.subplot(1, 2, 2)                                                                                
g = sns.distplot(data_num['tot_cur_bal']).set_title("After outlier treatement")


# In[209]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['tot_cur_bal'], y=data_num['default_ind'], fit_reg=False).set_title("After outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['tot_cur_bal']).set_title("After outlier treatement")
plt.show()


# In[210]:


data_num['tot_cur_bal_bb'] = pd.cut(data_num['tot_cur_bal'], 3)
data_num['tot_cur_bal_bb'].value_counts()


# In[211]:


data_num.loc[data_num['tot_cur_bal'] <= 139092.667, 'tot_cur_bal'] = 0
data_num.loc[data_num['tot_cur_bal'] > 139092.667, 'tot_cur_bal'] = 1
data_num['tot_cur_bal'].value_counts()


# In[212]:


sns.countplot(x='tot_cur_bal',hue='default_ind',data=data_num)
plt.tight_layout()


# In[213]:


data_num['installment']=data['installment']


# In[214]:


#installment
data_num['installment'].describe()


# In[215]:


#Checking Outlier
plt.subplots(figsize=(16, 5))
plt.figure(1)
plt.subplot(121)
g = sns.regplot(x=data_num['installment'], y=data_num['default_ind'], fit_reg=False).set_title("Before outlier treatement")
plt.subplot(122)
sns.boxplot(data=data_num,x=data_num['installment']).set_title("Before outlier treatement")
plt.show()


# In[216]:


plt.subplots(figsize=(16, 5))
sns.distplot(data_num['installment'])
plt.show() #outliers does not exist


# In[217]:


data_num['installment_bann'] = pd.cut(data_num['installment'], 3)
data_num['installment_bann'].value_counts()


# In[218]:


data_num.loc[data_num['installment'] <= 474.263, 'installment'] = 0
data_num.loc[data_num['installment'] > 474.263, 'installment'] = 1
data_num['installment'].value_counts()


# In[219]:


sns.countplot(x='installment',hue='default_ind',data=data_num)
plt.tight_layout()


# In[220]:


data_num.columns


# In[221]:


data_num.drop(['annual_inc_band', 'loan_amnt_band',
       'int_rate_band', 'dti_band', 'delinq_2yrs_bandvx',
       'inq_last_6mths_band', 'mths_since_last_delinq_band', 'open_acc_ba',
       'pub_rec_ban', 'revol_bal_baad', 'revol_util_ban', 'total_acc_band',
       'out_prncp_ban', 'total_pymnt_ban', 'total_rec_int_band',
       'total_rec_late_fee_band', 'recoveries_band', 'last_pymnt_amnt_ban',
       'collections_12_mths_ex_med_band', 'tot_coll_amt_band',
       'tot_cur_bal_bb', 'installment_bann'], axis = 1, inplace = True)


# In[222]:


print(data.shape, data_num.shape)

Created bands were dropped from Numerical data
# ### EDA will be performed on Categorical data with data type as object

# In[223]:


data_cat = data.select_dtypes(include = ['object'])


# In[224]:


data_cat.shape


# In[225]:


data_cat.isnull().sum()


# In[226]:


data_cat['default_ind'] = data['default_ind']


# In[227]:


data_cat.columns


# In[228]:


#grade
data_cat['grade'].value_counts()


# In[229]:


grade_map={'A':1, 'B':2, 'C':3, 'D': 4, 'E':5, 'F':5, 'G':5}
data_cat['grade']=data_cat['grade'].map(grade_map)


# In[230]:


data['grade']=data['grade'].map(grade_map)


# In[231]:


data_cat['grade'].value_counts()


# In[232]:


plt.figure(figsize=(7,5))
sns.countplot(x='grade',hue='default_ind',data=data_cat)
plt.tight_layout()


# In[233]:


#sub_grade
data_cat['sub_grade'].value_counts()  # A lot of data exists will keep this for categorical


# In[234]:


#home_ownership
data_cat['home_ownership'].value_counts()


# In[235]:


home_map={'MORTGAGE':1, 'RENT':2, 'OWN':3, 'OTHER':4, 'NONE':4, 'ANY':4}
data_cat['home_ownership']=data_cat['home_ownership'].map(home_map)
data['home_ownership']=data['home_ownership'].map(home_map)
data_cat['home_ownership'].value_counts()


# In[236]:


plt.figure(figsize=(8,5))
sns.countplot(x='home_ownership',hue='default_ind',data=data_cat)
plt.tight_layout()


# In[237]:


#verification_status
data_cat['verification_status'].value_counts()


# In[238]:


verify_map={'Source Verified':1, 'Not Verified':2, 'Verified':3}
data_cat['verification_status']=data_cat['verification_status'].map(verify_map)
data['verification_status']=data['verification_status'].map(verify_map)
data_cat['verification_status'].value_counts()


# In[239]:


plt.figure(figsize=(8,5))
sns.countplot(x='verification_status',hue='default_ind',data=data_cat)
plt.tight_layout()


# In[240]:


#purpose
data_cat['purpose'].value_counts()  # part of categorical data so numerical mapping wont be done


# In[241]:


plt.figure(figsize=(15,10))
sns.countplot(x='purpose',hue='default_ind',data=data_cat)
plt.tight_layout()


# In[242]:


#term  ##bcategorical dummyfying needs to be done
data_cat['term'].value_counts()


# In[243]:


plt.figure(figsize=(8,6))
sns.countplot(x='term',hue='default_ind',data=data_cat)
plt.tight_layout()


# In[244]:


#application_type
data_cat['application_type'].value_counts() 


# In[245]:


app_map={'INDIVIDUAL':1, 'JOINT':2}
data_cat['application_type']=data_cat['application_type'].map(app_map)
data['application_type']=data['application_type'].map(app_map)
data_cat['application_type'].value_counts()


# In[246]:


plt.figure(figsize=(8,5))
sns.countplot(x='application_type',hue='default_ind',data=data_cat)
plt.tight_layout()


# In[247]:


#zip_code
data_cat['zip_code'].describe()  # does not add value so it can be dropped, 931 unique value exists which cant be treated


# In[248]:


#pymnt_plan
data_cat['pymnt_plan'].describe()   #doesnot convey much can be dropped


# In[249]:


#title
data_cat['title'].describe()   #to be dropped as a lot of categories exists ie 60840 unique values


# In[250]:


#addr_state
data_cat['addr_state'].describe() # has to dummyfied


# In[251]:


plt.figure(figsize=(16,8))
sns.countplot(x='addr_state',hue='default_ind',data=data_cat)
plt.tight_layout()


# In[252]:


#earliest_cr_line
data_cat['earliest_cr_line'].describe() #Categorical data


# In[253]:


#initial_list_status
data_cat['initial_list_status'].value_counts()


# In[254]:


list_status_map={'f':1, 'w':2}
data_cat['initial_list_status']=data_cat['initial_list_status'].map(list_status_map)


# In[255]:


data['initial_list_status']=data['initial_list_status'].map(list_status_map)
data_cat['initial_list_status'].value_counts()


# In[256]:


plt.figure(figsize=(8,5))
sns.countplot(x='initial_list_status',hue='default_ind',data=data_cat)
plt.tight_layout()  


# In[257]:


#last_pymnt_d
data_cat['last_pymnt_d'].describe()  # has to dummyfied a lot of categiries


# In[258]:


#next_pymnt_d 
data_cat['next_pymnt_d'].value_counts()


# In[259]:


next_pymnt_status_map={'Feb-2016':1, 'Jan-2016':2, 'Mar-2016': 3}
data_cat['next_pymnt_d']=data_cat['next_pymnt_d'].map(next_pymnt_status_map)
data['next_pymnt_d']=data['next_pymnt_d'].map(next_pymnt_status_map)
data_cat['next_pymnt_d'].value_counts()


# In[260]:


plt.figure(figsize=(8,5))
sns.countplot(x='next_pymnt_d',hue='default_ind',data=data_cat)
plt.tight_layout()  


# In[261]:


#last_credit_pull_d 
data_cat['last_credit_pull_d'].value_counts() # categorical data- has to be dummyfied


# In[262]:


data_cat['emp_title'].describe() # too many categories so it has to be dropped, 289475 unique values.


# In[263]:


#emp_length
data_cat['emp_length'].value_counts()


# In[264]:


data_cat['emp_length'].isnull().sum()


# In[265]:


#mapping and null value treatement
emp_range= {'< 1 year':0.5, '1 year':1, '2 years': 2, '3 years':3,
            '4 years':4, '5 years':5,'6 years':6,'7 years':7,
            '8 years':8,'9 years':9, '10+ years':10}
data_cat['emp_length'] = data_cat["emp_length"].map(emp_range)
nullseries=pd.isnull(data_cat).sum()
nullseries[nullseries>0]
data_cat['emplen'] = data_cat['emp_length'].replace(np.nan, 10)
data_cat.drop(['emp_length'],axis=1,inplace=True)
data_cat['emplen'].value_counts() #category


# In[266]:


data['emplen']=data_cat['emplen']


# In[267]:


#To reset all the indexes 
data.reset_index(drop=True, inplace=True)
data_cat.reset_index(drop=True, inplace=True)
data_num.reset_index(drop=True, inplace=True)


# In[268]:


data_cat.columns


# In[269]:


data_ordinal=data_cat.loc[:,['term','sub_grade','purpose','application_type','addr_state', 
                             'last_pymnt_d', 'last_credit_pull_d']] 


# In[270]:


data_ordinal.columns


# In[271]:


print(data.shape,data_num.shape,data_cat.shape,data_ordinal.shape)


# In[272]:


data_numerical = data_cat.loc[:,['grade','home_ownership','verification_status','initial_list_status','next_pymnt_d','emplen',
                                'application_type']]


# In[273]:


data_numerical.shape


# In[274]:


data_num_all = pd.concat([data_num,data_numerical], axis = 1)


# In[275]:


data_num_all.shape


# In[276]:


data_num_all.drop(['default_ind'], axis =1, inplace = True)


# In[277]:


data_num_all.columns # data_num_all is basically combination of al Pd.cut data performed on data set


# #### Now We will encode Categorical columns

# In[278]:


from sklearn.preprocessing import LabelEncoder


# In[279]:


LabelEncoder_categorical = data_cat.loc[:,['term','sub_grade','purpose','addr_state', 
                             'last_pymnt_d', 'last_credit_pull_d']]
dummyEncoder_categorical = data_cat.loc[:,['term','sub_grade','purpose','addr_state', 
                             'last_pymnt_d', 'last_credit_pull_d']]


# In[280]:


categorical_feature_mask = LabelEncoder_categorical.dtypes==object
categorical_feature_mask


# In[281]:


categorical_cols = LabelEncoder_categorical.columns[categorical_feature_mask].tolist()


# In[282]:


le = LabelEncoder()
LabelEncoder_categorical[categorical_cols] = LabelEncoder_categorical[categorical_cols].apply(lambda col: le.fit_transform(col))
LabelEncoder_categorical.head()


# In[283]:


LabelEncoder_categorical.shape


# In[284]:


dummyEncoder_categorical = pd.get_dummies(dummyEncoder_categorical, drop_first=True)
dummyEncoder_categorical.shape


# In[285]:


data_num.columns


# In[286]:


data_numerical.columns


# In[287]:


scaled_all_numeric= data.loc[:,['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs',
       'inq_last_6mths', 'mths_since_last_delinq', 'open_acc', 'pub_rec',
       'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'total_pymnt',
       'total_rec_int', 'total_rec_late_fee', 'recoveries', 'last_pymnt_amnt',
       'collections_12_mths_ex_med', 'acc_now_delinq', 'tot_coll_amt',
       'tot_cur_bal', 'installment', 'grade', 'home_ownership', 'verification_status', 'initial_list_status',
       'next_pymnt_d', 'emplen','application_type' ]]


# #### We will Scale all the numerical data in dataset

# In[288]:


from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(scaled_all_numeric)
scaled_numeric_all = scaler.transform(scaled_all_numeric)
Numeric_Scaled = pd.DataFrame(scaled_numeric_all, columns=scaled_all_numeric.columns.tolist())


# In[289]:


com1 = pd.concat([Numeric_Scaled,dummyEncoder_categorical], axis=1)


# In[290]:


print(Numeric_Scaled.shape, dummyEncoder_categorical.shape, com1.shape)


# ##### We will split data into train & test based on issue_d column as per problem statement

# In[291]:


com1[['issue_d','default_ind']]= data_cat[['issue_d','default_ind']]
#for splitting purpose'
com1['issue_d'] = pd.to_datetime(com1['issue_d'])
# Creating train and test data set According to problem statement given The data should be divided into train 
# June 2007 - May 2015 )'and out-of-time test ( June 2015 - Dec 2015 ) data.'

train1 = com1[com1['issue_d'] < '2015-6-01']
test1 = com1[com1['issue_d'] >= '2015-6-01']
print(train1.shape, test1.shape)


# In[292]:


train1 =train1.drop('issue_d' , axis=1)
test1 =test1.drop('issue_d', axis=1)
"""X = train1.iloc[:,0:-1]
y = train1['default_ind']"""

X_train1 = train1.iloc[:,0:-1]
y_train1 = train1['default_ind']
X_test_com1  = test1.iloc[:,0:-1]
y_test_com1  = test1['default_ind']


# In[293]:


print(X_train1.shape, y_train1.shape, X_test_com1.shape, y_test_com1.shape)


# #### We will split the dataframe into Subtrain & validation so as to reduce training time for model 

# In[294]:


from sklearn.model_selection import train_test_split 
X_train_com1, X_val1, y_train_com1, y_val1 = train_test_split(X_train1, y_train1, test_size = 0.3, random_state = 0)
print(X_train_com1.shape, y_train_com1.shape, X_val1.shape, y_val1.shape)


# ### To treat data imbalance we will use SMOTE on train dataframe

# In[295]:


from imblearn.over_sampling import SMOTE 
sm1 = SMOTE(random_state = 2) 
X_sm1, y_sm1 = sm1.fit_sample(X_train_com1, y_train_com1.ravel()) 


# In[296]:


print('Before OverSampling, X: {}'.format(X_train_com1.shape)) 
print('Before OverSampling, y: {}'.format(y_train_com1.shape)) 
print("Before OverSampling, counts of '1': {}".format(sum(y_train_com1 == 1))) 
print("Before OverSampling, counts of '0': {}".format(sum(y_train_com1 == 0))) 
print('\n')
print('With imbalance treatment:'.upper())
print('After OverSampling, X: {}'.format(X_sm1.shape)) 
print('After OverSampling, y: {}'.format(y_sm1.shape)) 
print("After OverSampling, counts of '1': {}".format(sum(y_sm1 == 1))) 
print("After OverSampling, counts of '0': {}".format(sum(y_sm1 == 0))) 
print('\n')


# #### because of highly unbalanced data lets apply SMOTE on test as well to verify

# In[335]:


from imblearn.over_sampling import SMOTE 
smt = SMOTE(random_state = 2) 
X_smt1, y_smt1 = smt.fit_sample(X_test_com1, y_test_com1.ravel()) 


# In[336]:


print('Before OverSampling, X: {}'.format(X_test_com1.shape)) 
print('Before OverSampling, y: {}'.format(y_test_com1.shape)) 
print("Before OverSampling, counts of '1': {}".format(sum(y_test_com1 == 1))) 
print("Before OverSampling, counts of '0': {}".format(sum(y_test_com1 == 0))) 
print('\n')
print('With imbalance treatment:'.upper())
print('After OverSampling, X: {}'.format(X_smt1.shape)) 
print('After OverSampling, y: {}'.format(y_smt1.shape)) 
print("After OverSampling, counts of '1': {}".format(sum(y_smt1 == 1))) 
print("After OverSampling, counts of '0': {}".format(sum(y_smt1 == 0))) 
print('\n')


# ### For imbalance data treatement Lets use UnderSampler

# In[351]:


from imblearn.under_sampling import (RandomUnderSampler)
un = RandomUnderSampler(random_state=2) 
X_sm3, y_sm3 = un.fit_sample(X_train_com1, y_train_com1.ravel()) 


# In[353]:


print('Before Undersampling, X: {}'.format(X_train_com1.shape)) 
print('Before Undersampling, y: {}'.format(y_train_com1.shape)) 
print("Before Undersampling, counts of '1': {}".format(sum(y_train_com1 == 1))) 
print("Before Undersampling, counts of '0': {}".format(sum(y_train_com1 == 0))) 
print('\n')
print('With imbalance treatment:'.upper())
print('After Undersampling, X: {}'.format(X_sm3.shape)) 
print('After Undersampling, y: {}'.format(y_sm3.shape)) 
print("After Undersampling, counts of '1': {}".format(sum(y_sm3 == 1))) 
print("After Undersampling, counts of '0': {}".format(sum(y_sm3 == 0))) 
print('\n')


# # MODEL FITTING

# ## LOGISTIC REGRESSION

# ### LOGISTIC REGRESSION - Without SMOTE

# In[354]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, classification_report , f1_score, accuracy_score, roc_auc_score
from confusionMatrix import plotConfusionMatrix

lR = LogisticRegression(max_iter=200, C=0.5)
lR.fit(X_train_com1, y_train_com1)
acc_v=lR.score(X_val1, y_val1)
acc = lR.score(X_test_com1, y_test_com1)
predv= lR.predict(X_val1)
preds = lR.predict(X_test_com1)
pred_probav = lR.predict_proba(X_val1)[::,1]
pred_proba = lR.predict_proba(X_test_com1)[::,1]
print('*'*80)
print('Logistic Regression:')
print("Accuracy without SNOTE on validation set: %.2f%%" % (acc_v * 100.0))
print("Accuracy without SNOTE on test set: %.2f%%" % (acc * 100.0))
print('*'*80)
print('F1 score val:\n', classification_report(y_val1, predv)) 
#print('\n')
print('*'*80)
print('F1 score test:\n', classification_report(y_test_com1, preds)) 
print('*'*80)
fpr_G, tpr_G, _G = roc_curve(y_test_com1,  pred_proba)
aucv = roc_auc_score(y_val1, pred_probav)
plt.figure(figsize = (6,5))
plt.plot(fpr_G,tpr_G,label="LG w/o SMOTE on val, area="+str(np.round(aucv,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.style.use('default')
plt.show()
print('*'*80)


# In[355]:


fpr_G, tpr_G, _G = roc_curve(y_test_com1,  pred_proba)
auc = roc_auc_score(y_test_com1, pred_proba)
plt.figure(figsize = (6,5))
plt.plot(fpr_G,tpr_G,label="LG w/o SMOTE on test, area="+str(np.round(auc,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.style.use('default')
plt.style.context('dark_background')
plt.show()

print('*'*80)
cnf_mat = confusion_matrix(y_test_com1,preds)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))


# #### LOGISTIC REGRESSION - With SMOTE

# In[299]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, classification_report , f1_score, accuracy_score, roc_auc_score
from confusionMatrix import plotConfusionMatrix
lR = LogisticRegression(max_iter=200, C=0.5)
lR.fit(X_sm1, y_sm1)
lR.score(X_sm1, y_sm1)
acc = lR.score(X_test_com1, y_test_com1)
preds = lR.predict(X_test_com1)
pred_proba = lR.predict_proba(X_test_com1)[::,1]
print('*'*80)
print('Logistic Regression:')
print("Accuracy with SMOTE: %.2f%%" % (acc * 100.0))
print('*'*80)

print(classification_report(y_test_com1, preds)) 
print('*'*80)
#y_pred_proba_G = model_G.predict_proba(X_val_res)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_test_com1,  pred_proba)
auc = roc_auc_score(y_test_com1, pred_proba)
plt.figure(figsize = (6,5))
plt.plot(fpr_G,tpr_G,label="LF With SMOTE,Area="+str(np.round(auc,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)

print(confusion_matrix(y_test_com1, preds))
cnf_mat = confusion_matrix(y_test_com1, preds)
f1= f1_score(y_test_com1,preds, average='micro')
plt.figure()
print('*'*80)
cnf_mat = confusion_matrix(y_test_com1,preds)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))


# ### Logistic Regression with SMOTE on test as well

# In[337]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, classification_report , f1_score, accuracy_score, roc_auc_score
from confusionMatrix import plotConfusionMatrix
lR = LogisticRegression(max_iter=200, C=0.5)
lR.fit(X_sm1, y_sm1)
lR.score(X_sm1, y_sm1)
acct = lR.score(X_smt1, y_smt1)
preds = lR.predict(X_smt1)
pred_proba = lR.predict_proba(X_smt1)[::,1]
print('*'*80)
print('Logistic Regression:')
print("Accuracy with SMOTE: %.2f%%" % (acct * 100.0))
print('*'*80)

print(classification_report(y_smt1, preds)) 
print('*'*80)
#y_pred_proba_G = model_G.predict_proba(X_val_res)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_smt1,  pred_proba)
auct = roc_auc_score(y_smt1, pred_proba)
plt.figure(figsize = (6,5))
plt.plot(fpr_G,tpr_G,label="LF With SMOTE,Area="+str(np.round(auct,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)

print(confusion_matrix(y_smt1, preds))
cnf_mat = confusion_matrix(y_smt1, preds)
f1= f1_score(y_smt1,preds, average='micro')
plt.figure()
print('*'*80)
cnf_mat = confusion_matrix(y_smt1,preds)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))


# ### Logistic regression with undersampling of majority

# In[377]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, classification_report , f1_score, accuracy_score, roc_auc_score
from confusionMatrix import plotConfusionMatrix
lR = LogisticRegression(max_iter=200, C=0.5)
lR.fit(X_sm3, y_sm3)
lR.score(X_sm3, y_sm3)
acc = lR.score(X_test_com1, y_test_com1)
preds = lR.predict(X_test_com1)
pred_proba = lR.predict_proba(X_test_com1)[::,1]
print('*'*80)
print('Logistic Regression:')
print("Accuracy with UnderSampling: %.2f%%" % (acc * 100.0))
print('*'*80)

print(classification_report(y_test_com1, preds)) 
print('*'*80)
#y_pred_proba_G = model_G.predict_proba(X_val_res)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_test_com1,  pred_proba)
auc = roc_auc_score(y_test_com1, pred_proba)
plt.figure(figsize = (6,5))
plt.plot(fpr_G,tpr_G,label="LF With UnderSampling,Area="+str(np.round(auc,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)

print(confusion_matrix(y_test_com1, preds))
cnf_mat = confusion_matrix(y_test_com1, preds)
f1= f1_score(y_test_com1,preds, average='micro')
plt.figure()
print('*'*80)
cnf_mat = confusion_matrix(y_test_com1,preds)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))


# ## Decision Tree

# ### Decision Tree - Without SMOTE

# In[306]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report , f1_score, accuracy_score, roc_auc_score, roc_curve
from confusionMatrix import plotConfusionMatrix
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train_com1, y_train_com1.ravel()) 
accv = clf.score(X_val1, y_val1)
acc = clf.score(X_test_com1, y_test_com1)
predictions_v = clf.predict(X_val1)
predictions_ = clf.predict(X_test_com1) 
y_pred_probav = clf.predict_proba(X_val1)[::,1]
y_pred_proba = clf.predict_proba(X_test_com1)[::,1]
print('*'*80)
print('Decision Tree:')
print("Accuracy without SMOTE on val: %.2f%%" % (accv * 100.0))
print("Accuracy without SMOTE on test: %.2f%%" % (acc * 100.0))

print('*'*80)
print(confusion_matrix(y_val1, predictions_v))
# print classification report 
print('Without SMOTE on Val:'.upper())
print(classification_report(y_val1, predictions_v)) 
print('*'*80)
print(confusion_matrix(y_test_com1, predictions_))
# print classification report 
print('Without SMOTE on Test:'.upper())
print(classification_report(y_test_com1, predictions_)) 
print('*'*80)
#print('\n')


# In[308]:


## ROC curve
y_pred_probav = clf.predict_proba(X_val1)[::,1]
fpr, tpr, _ = roc_curve(y_test_com1,  y_pred_probav)
aucv = roc_auc_score(y_val1, y_pred_probav)
plt.plot(fpr,tpr,label="DT without SMOTE on val, Area="+str(np.round(aucv,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)

## ROC curve
y_pred_proba = clf.predict_proba(X_test_com1)[::,1]
fpr, tpr, _ = roc_curve(y_test_com1,  y_pred_proba)
auc = roc_auc_score(y_test_com1, y_pred_proba)
plt.plot(fpr,tpr,label="DT without SMOTE, Area="+str(np.round(auc,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)
cnf_mat = confusion_matrix(y_test_com1,predictions_)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)


# #### Decision Tree - With SMOTE

# In[311]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report , f1_score, accuracy_score, roc_auc_score, roc_curve
from confusionMatrix import plotConfusionMatrix
clfs = DecisionTreeClassifier(random_state=0)
clfs.fit(X_sm1, y_sm1.ravel()) 
acc = clfs.score(X_test_com1, y_test_com1)
predictions_ = clfs.predict(X_test_com1) 
y_pred_proba = clfs.predict_proba(X_test_com1)[::,1]
print('*'*80)
print('Decision Tree:')
print("Accuracy with SMOTE: %.2f%%" % (acc * 100.0))
print('*'*80)
print(confusion_matrix(y_test_com1, predictions_))
print('*'*80)
# print classification report 
print('With SMOTE:'.upper())
print(classification_report(y_test_com1, predictions_)) 
print('*'*80)
#print('\n')
## ROC curve
y_pred_proba = clfs.predict_proba(X_test_com1)[::,1]
fpr, tpr, _ = roc_curve(y_test_com1,  y_pred_proba)
auc = roc_auc_score(y_test_com1, y_pred_proba)
plt.plot(fpr,tpr,label="Gs-DT with SMOTE, auc="+str(np.round(auc,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)
cnf_mat = confusion_matrix(y_test_com1,predictions_)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)


# #### Decision Tree With SMOTE on Test data

# In[339]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report , f1_score, accuracy_score, roc_auc_score, roc_curve
from confusionMatrix import plotConfusionMatrix
clfs = DecisionTreeClassifier(random_state=0)
clfs.fit(X_sm1, y_sm1.ravel()) 
acc = clfs.score(X_smt1, y_smt1)
predictions_ = clfs.predict(X_smt1) 
y_pred_proba = clfs.predict_proba(X_smt1)[::,1]
print('*'*80)
print('Decision Tree:')
print("Accuracy with SMOTE: %.2f%%" % (acc * 100.0))
print('*'*80)
print(confusion_matrix(y_smt1, predictions_))
print('*'*80)
# print classification report 
print('With SMOTE:'.upper())
print(classification_report(y_smt1, predictions_)) 
print('*'*80)
#print('\n')
## ROC curve
y_pred_proba = clfs.predict_proba(X_smt1)[::,1]
fpr, tpr, _ = roc_curve(y_smt1,  y_pred_proba)
auc = roc_auc_score(y_smt1, y_pred_proba)
plt.plot(fpr,tpr,label="Gs-DT with SMOTE, auc="+str(np.round(auc,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)
cnf_mat = confusion_matrix(y_smt1,predictions_)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)


# ### Decision Tree with UnderSampling on Train

# In[357]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report , f1_score, accuracy_score, roc_auc_score, roc_curve
from confusionMatrix import plotConfusionMatrix
clfs = DecisionTreeClassifier(random_state=0)
clfs.fit(X_sm3, y_sm3.ravel()) 
acc = clfs.score(X_test_com1, y_test_com1)
predictions_ = clfs.predict(X_test_com1) 
y_pred_proba = clfs.predict_proba(X_test_com1)[::,1]
print('*'*80)
print('Decision Tree:')
print("Accuracy with UnderSampling: %.2f%%" % (acc * 100.0))
print('*'*80)
print(confusion_matrix(y_test_com1, predictions_))
print('*'*80)
# print classification report 
print('With SMOTE:'.upper())
print(classification_report(y_test_com1, predictions_)) 
print('*'*80)
#print('\n')
## ROC curve
y_pred_proba = clfs.predict_proba(X_test_com1)[::,1]
fpr, tpr, _ = roc_curve(y_test_com1,  y_pred_proba)
auc = roc_auc_score(y_test_com1, y_pred_proba)
plt.plot(fpr,tpr,label="Gs-DT with SMOTE, auc="+str(np.round(auc,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)
cnf_mat = confusion_matrix(y_test_com1,predictions_)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)


# #### GRID SEARCH WITH SMOTE

# In[302]:


from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import metrics

parameters = {'max_depth': np.arange(3, 10)} # pruning
tree = GridSearchCV(clfs,parameters)
tree.fit(X_sm1,y_sm1)
preds = tree.predict(X_test_com1)
accu = tree.score(X_test_com1, y_test_com1)

print('GRID SEARCH WITH SMOTE -- DT:')
print('Using best parameters:',tree.best_params_)
print("Accuracy with SMOTE & Grid Search: %.2f%%" % (acc * 100.0))

y_pred_proba_ = tree.predict_proba(X_test_com1)[::,1]
fpr, tpr, _ = roc_curve(y_test_com1,  y_pred_proba_)
auc = roc_auc_score(y_test_com1, y_pred_proba_)
plt.plot(fpr,tpr,label="Gs-Smote-DT, Area="+str(np.round(auc,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)


# ### GRID SEARCH WITH SMOTE & CROSS VALIDATION

# In[313]:


def dtree_grid_search(X,y,nfolds):
    #create a dictionary of all values we want to test
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(15, 30)}
    # decision tree model
    dtree_model = DecisionTreeClassifier()
    #use gridsearch to val all values
    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=nfolds)
    #fit model to data
    dtree_gscv.fit(X, y)
    #find score
    score = dtree_gscv.score(X, y)
    
    return dtree_gscv.best_params_, score, dtree_gscv

print('GRID SEARCH WITH SMOTE & CROSS VALIDATION -- DT:')
best_param, acc, model = dtree_grid_search(X_sm1,y_sm1, 4)
preds = model.predict(X_test_com1)
acc = model.score(X_test_com1, y_test_com1)
print('Using best parameters:',best_param)
print("Accuracy with SMOTE & Grid Search & CV: %.2f%%" % (acc * 100.0))
print('*'*80)
print(classification_report(y_test_com1, preds)) 
print(confusion_matrix(y_test_com1, preds))
print('*'*80)
## ROC curve
y_pred_proba = model.predict_proba(X_test_com1)[::,1]
fpr, tpr, _ = roc_curve(y_test_com1,  y_pred_proba)
auc = metrics.roc_auc_score(y_test_com1, y_pred_proba)
plt.plot(fpr,tpr,label="Gs-Smote-cv-DT, Area="+str(np.round(auc,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)

cnf_mat = confusion_matrix(y_test_com1,preds)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)


# ## RANDOM FOREST

# ### Random Forest without SMOTE

# In[358]:


from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
model.fit(X_train_com1, y_train_com1)
print('*'*80)
print('Random Forest:')
# Actual class predictions
probs = model.predict(X_test_com1)
probsv = model.predict(X_val1)
# Probabilities for each class
rf_probsv = model.predict_proba(X_val1)[:, 1]
rf_probs = model.predict_proba(X_test_com1)[:, 1]
acc_rfv = model.score(X_val1, y_val1)
acc_rf = model.score(X_test_com1, y_test_com1)
print("Accuracy with Random forest without SMOTE on val: %.2f%%" % (acc_rfv * 100.0))
print("Accuracy with Random forest without SMOTE on test: %.2f%%" % (acc_rf * 100.0))

print('*'*80)
print("Classification matrix on Val")
print(classification_report(y_val1, probsv)) 
print('*'*80)
print("Classification matrix on Val")
print(classification_report(y_test_com1, probs))
print('*'*80)
## ROC curve
fpr_rf, tpr_rf, _rf = roc_curve(y_val1,  rf_probsv)
auc_rfv = metrics.roc_auc_score(y_val1, rf_probsv)
plt.plot(fpr_rf,tpr_rf,label="RF w/o SMOTE on val, Area="+str(np.round(auc_rfv,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)


# In[359]:


## ROC curve
fpr_rf, tpr_rf, _rf = roc_curve(y_test_com1,  rf_probs)
auc_rf = metrics.roc_auc_score(y_test_com1, rf_probs)
plt.plot(fpr_rf,tpr_rf,label="RF w/o SMOTE on test, Area="+str(np.round(auc_rf,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)
print(confusion_matrix(y_test_com1, probs))
f1= f1_score(y_test_com1,probs, average='micro')
plt.figure()
print('*'*80)
cnf_mat = confusion_matrix(y_test_com1,probs)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)


# In[317]:


import pandas as pd

# Extract feature importances
fin_without_SMOTE = pd.DataFrame({'feature': list(X_train_com1.columns),
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending= False)

# Display
fin_without_SMOTE.head(5)


# ### Random Forest with SMOTE on train

# In[318]:


from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
model.fit(X_sm1, y_sm1)
print('*'*80)
print('Random Forest:')
# Actual class predictions
probs = model.predict(X_test_com1)
# Probabilities for each class
rf_probs = model.predict_proba(X_test_com1)[:, 1]
acc_rf = model.score(X_test_com1, y_test_com1)
print("Accuracy with Random forest: %.2f%%" % (acc * 100.0))
print('*'*80)
print(confusion_matrix(y_test_com1, probs))
print('*'*80)
print(classification_report(y_test_com1, probs)) 
print('*'*80)
## ROC curve
fpr_rf, tpr_rf, _rf = roc_curve(y_test_com1,  rf_probs)
auc_rf = metrics.roc_auc_score(y_test_com1, rf_probs)
plt.plot(fpr_rf,tpr_rf,label="RF with SMOTE, Area="+str(np.round(auc_rf,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)
cnf_mat = confusion_matrix(y_test_com1,probs)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)


# ### Random Forest with SMOTE on train & test

# In[340]:


from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
model.fit(X_sm1, y_sm1)
print('*'*80)
print('Random Forest:')
# Actual class predictions
probs = model.predict(X_smt1)
# Probabilities for each class
rf_probs = model.predict_proba(X_smt1)[:, 1]
acc_rf = model.score(X_smt1, y_smt1)
print("Accuracy with Random forest: %.2f%%" % (acc * 100.0))
print('*'*80)
print(confusion_matrix(y_smt1, probs))
print('*'*80)
print(classification_report(y_smt1, probs)) 
print('*'*80)
## ROC curve
fpr_rf, tpr_rf, _rf = roc_curve(y_smt1,  rf_probs)
auc_rf = metrics.roc_auc_score(y_smt1, rf_probs)
plt.plot(fpr_rf,tpr_rf,label="RF with SMOTE, Area="+str(np.round(auc_rf,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)
cnf_mat = confusion_matrix(y_smt1,probs)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)


# ### Random Forest with Undersampling of majority class

# In[360]:


from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
model.fit(X_sm3, y_sm3)
print('*'*80)
print('Random Forest:')
# Actual class predictions
probs = model.predict(X_test_com1)
# Probabilities for each class
rf_probs = model.predict_proba(X_test_com1)[:, 1]
acc_rf = model.score(X_test_com1, y_test_com1)
print("Accuracy with Random forest and undersampling: %.2f%%" % (acc * 100.0))
print('*'*80)
print(confusion_matrix(y_test_com1, probs))
print('*'*80)
print(classification_report(y_test_com1, probs)) 
print('*'*80)
## ROC curve
fpr_rf, tpr_rf, _rf = roc_curve(y_test_com1,  rf_probs)
auc_rf = metrics.roc_auc_score(y_test_com1, rf_probs)
plt.plot(fpr_rf,tpr_rf,label="RF with Undersampling, Area="+str(np.round(auc_rf,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)
cnf_mat = confusion_matrix(y_test_com1,probs)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)


# In[342]:


# Extract feature importances
finnimp_with_SMOTE = pd.DataFrame({'feature': list(X_sm1.columns),
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending= False)

# Display
finnimp_with_SMOTE.head(5)


# ## XG BOOST

# ### XG Boost Without SMOTE

# In[363]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#instantiate model and train
model_G = XGBClassifier()
model_G.fit(X_train_com1, y_train_com1)
#learning_rate =0.01, n_estimators=5000, max_depth=4,min_child_weight=6, gamma=0, subsample=0.8,colsample_bytree=0.8,
#reg_alpha=0.005, objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27


# In[364]:


y_pred = model_G.predict(X_test_com1)
y_predv = model_G.predict(X_val1)
preds = [round(value) for value in y_pred]
accGv = model_G.score(X_val1,y_val1)
accG = model_G.score(X_test_com1,y_test_com1)

print('*'*80)
print('XG Bosst:')
print("Accuracy without SMOTE on VAl: %.2f%%" % (accGv * 100.0))
print("Accuracy without SMOTE on Test: %.2f%%" % (accG * 100.0))
print('*'*80)

print("Calssification report for validation")
print(classification_report(y_val1, y_predv))
print('*'*80)
print("Calssification report for test")
print(classification_report(y_test_com1, y_pred))
print('*'*80)


# In[365]:


y_pred_proba_Gv = model_G.predict_proba(X_val1)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_val1,  y_pred_proba_Gv)
auc_Gv = metrics.roc_auc_score(y_val1, y_pred_proba_Gv)
plt.plot(fpr_G,tpr_G,label="XG Boost w/o SNOTE on val, Area="+str(np.round(auc_Gv,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)
y_pred_proba_G = model_G.predict_proba(X_test_com1)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_test_com1,  y_pred_proba_G)
auc_G = metrics.roc_auc_score(y_test_com1, y_pred_proba_G)
plt.plot(fpr_G,tpr_G,label="XG Boost w/o SNOTE on test, Area="+str(np.round(auc_G,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)
cnf_mat = confusion_matrix(y_test_com1,y_pred)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)


# In[322]:


from xgboost import plot_importance
# plot feature importance
plt.rcParams["figure.figsize"] = (14, 7)
plot_importance(model_G)


# ### XG Boost With SMOTE on train

# In[323]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#instantiate model and train
model_G = XGBClassifier()
model_G.fit(X_sm1, y_sm1)
#learning_rate =0.01, n_estimators=5000, max_depth=4,min_child_weight=6, gamma=0, subsample=0.8,colsample_bytree=0.8,
#reg_alpha=0.005, objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27


# In[324]:



y_pred = model_G.predict(X_test_com1)
preds = [round(value) for value in y_pred]

accG = model_G.score(X_test_com1,y_test_com1)

print('*'*80)
print('XG Bosst:')
print("Accuracy with SMOTE: %.2f%%" % (accG * 100.0))
print('*'*80)
print(classification_report(y_test_com1, y_pred)) 
print('*'*80)
print(confusion_matrix(y_test_com1, y_pred))
print('*'*80)
y_pred_proba_G = model_G.predict_proba(X_test_com1)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_test_com1,  y_pred_proba_G)
auc_G = metrics.roc_auc_score(y_test_com1, y_pred_proba_G)
plt.plot(fpr_G,tpr_G,label="XGB with SMOTE, Area="+str(np.round(auc_G,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)
cnf_mat = confusion_matrix(y_test_com1,y_pred)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)


# ### XG Boost with SMOTE on test as well

# In[343]:



y_pred = model_G.predict(X_smt1)
preds = [round(value) for value in y_pred]

accG = model_G.score(X_smt1,y_smt1)

print('*'*80)
print('XG Bosst:')
print("Accuracy with SMOTE: %.2f%%" % (accG * 100.0))
print('*'*80)
print(classification_report(y_smt1, y_pred)) 
print('*'*80)
print(confusion_matrix(y_smt1, y_pred))
print('*'*80)
y_pred_proba_G = model_G.predict_proba(X_smt1)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_smt1,  y_pred_proba_G)
auc_G = metrics.roc_auc_score(y_smt1, y_pred_proba_G)
plt.plot(fpr_G,tpr_G,label="XGB with SMOTE, Area="+str(np.round(auc_G,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)
cnf_mat = confusion_matrix(y_smt1,y_pred)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)


# ### XG Boost with UnderSampling on Train

# In[367]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#instantiate model and train
model_G = XGBClassifier()
model_G.fit(X_sm3, y_sm3)
#learning_rate =0.01, n_estimators=5000, max_depth=4,min_child_weight=6, gamma=0, subsample=0.8,colsample_bytree=0.8,
#reg_alpha=0.005, objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27


y_pred = model_G.predict(X_test_com1)
preds = [round(value) for value in y_pred]

accG = model_G.score(X_test_com1,y_test_com1)

print('*'*80)
print('XG Bosst:')
print("Accuracy with UnderSampling: %.2f%%" % (accG * 100.0))
print('*'*80)
print(classification_report(y_test_com1, y_pred)) 
print('*'*80)
print(confusion_matrix(y_test_com1, y_pred))
print('*'*80)
y_pred_proba_G = model_G.predict_proba(X_test_com1)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_test_com1,  y_pred_proba_G)
auc_G = metrics.roc_auc_score(y_test_com1, y_pred_proba_G)
plt.plot(fpr_G,tpr_G,label="XGB with Undersampling, Area="+str(np.round(auc_G,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)
cnf_mat = confusion_matrix(y_test_com1,y_pred)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)


# In[325]:


from xgboost import plot_importance
# plot feature importance
plt.rcParams["figure.figsize"] = (14, 7)
plot_importance(model_G)


# ## SVM

# ### SVM Without SMOTE

# In[326]:


com1.shape


# In[327]:


random_subset = com1.sample(n=85000)


# In[328]:


#random_subset['issue_d'] = pd.to_datetime(random_subset['issue_d'])
tr1 = random_subset[random_subset['issue_d'] < '2015-6-01']
te1 = random_subset[random_subset['issue_d'] >= '2015-6-01']
print(tr1.shape, te1.shape)


# In[329]:


tr1 =tr1.drop('issue_d' , axis=1)
te1 =te1.drop('issue_d', axis=1)

X_train11 = tr1.iloc[:,0:-1]
y_train11 = tr1['default_ind']
X_test_com11  = te1.iloc[:,0:-1]
y_test_com11  = te1['default_ind']
print(X_train11.shape, y_train11.shape, X_test_com11.shape, y_test_com11.shape)


# In[330]:


from sklearn.model_selection import train_test_split 
X_train_com11, X_val11, y_train_com11, y_val11 = train_test_split(X_train11, y_train11, test_size = 0.4, random_state = 0)
print(X_train_com11.shape, y_train_com11.shape, X_val11.shape, y_val11.shape)


# #### To treat data Imbalance applying SMOTE on Train Data

# In[331]:


from imblearn.over_sampling import SMOTE 
sm11 = SMOTE(random_state = 2) 
X_sm11, y_sm11 = sm11.fit_sample(X_train_com11, y_train_com11.ravel()) 

print('Before OverSampling, X: {}'.format(X_train_com11.shape)) 
print('Before OverSampling, y: {}'.format(y_train_com11.shape)) 
print("Before OverSampling, counts of '1': {}".format(sum(y_train_com11 == 1))) 
print("Before OverSampling, counts of '0': {}".format(sum(y_train_com11 == 0))) 
print('\n')
print('With imbalance treatment:'.upper())
print('After OverSampling, X: {}'.format(X_sm11.shape)) 
print('After OverSampling, y: {}'.format(y_sm11.shape)) 
print("After OverSampling, counts of '1': {}".format(sum(y_sm11 == 1))) 
print("After OverSampling, counts of '0': {}".format(sum(y_sm11 == 0))) 
print('\n')


# #### Since test is highly unbalanced lets do SMOTE also on test 

# In[345]:



from imblearn.over_sampling import SMOTE 
smt11 = SMOTE(random_state = 2) 
X_smt11, y_smt11 = smt11.fit_sample(X_test_com11, y_test_com11.ravel()) 

print('Before OverSampling, X: {}'.format(X_test_com11.shape)) 
print('Before OverSampling, y: {}'.format(y_test_com11.shape)) 
print("Before OverSampling, counts of '1': {}".format(sum(y_test_com11 == 1))) 
print("Before OverSampling, counts of '0': {}".format(sum(y_test_com11 == 0))) 
print('\n')
print('With imbalance treatment:'.upper())
print('After OverSampling, X: {}'.format(X_smt11.shape)) 
print('After OverSampling, y: {}'.format(y_smt11.shape)) 
print("After OverSampling, counts of '1': {}".format(sum(y_smt11 == 1))) 
print("After OverSampling, counts of '0': {}".format(sum(y_smt11 == 0))) 
print('\n')


# ### Random UnderSampler to treat data Imbalance

# In[368]:


from imblearn.under_sampling import (RandomUnderSampler)
un1 = RandomUnderSampler(random_state=2) 
X_sm12, y_sm12 = un1.fit_sample(X_train_com11, y_train_com11.ravel()) 

print('Before Undersampling, X: {}'.format(X_train_com11.shape)) 
print('Before Undersampling, y: {}'.format(y_train_com11.shape)) 
print("Before Undersampling, counts of '1': {}".format(sum(y_train_com11 == 1))) 
print("Before Undersampling, counts of '0': {}".format(sum(y_train_com11 == 0))) 
print('\n')
print('With imbalance treatment:'.upper())
print('After Undersampling, X: {}'.format(X_sm12.shape)) 
print('After Undersampling, y: {}'.format(y_sm12.shape)) 
print("After Undersampling, counts of '1': {}".format(sum(y_sm12 == 1))) 
print("After Undersampling, counts of '0': {}".format(sum(y_sm12 == 0))) 
print('\n')


# ### SVM without SMOTE

# In[369]:


from sklearn.metrics import confusion_matrix, classification_report, f1_score
from imblearn.over_sampling import SMOTE 
from sklearn.svm import SVC #Support vector classifier
from sklearn.metrics import roc_auc_score, roc_curve
from confusionMatrix import plotConfusionMatrix
svm = SVC() #SVC(kernel='linear') 

clf = svm.fit(X_train_com11, y_train_com11.ravel()) 
score1 = svm.score(X_val11, y_val11)
score2 = svm.score(X_test_com11, y_test_com11)
pred1 = svm.predict(X_val11) 
pred2 = svm.predict(X_test_com11)  
# print classification report 
print('Without SMOTE:'.upper())
print('Validation accuracy: ', score1)
print('test accuracy: ', score2)
print('*'*80)
print(confusion_matrix(y_test_com11, pred2))
print('*'*80)
print('F1 score val:\n', classification_report(y_val11, pred1)) 
print('*'*80)
print('F1 score test:\n', classification_report(y_test_com11, pred2)) 
print('*'*80)


# In[370]:


# ROC Curve
svm_p = SVC(probability = True) # for probability
clf_p = svm_p.fit(X_train_com11, y_train_com11.ravel()) # for probability
pred_proba = svm_p.predict_proba(X_test_com11)[::,1]
fpr, tpr, _ = roc_curve(y_test_com11,  pred_proba)
auc = roc_auc_score(y_test_com11, pred_proba)
plt.plot(fpr,tpr,label="SVM Without SMOTE, area="+str(np.round(auc,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)
cnf_mat = confusion_matrix(y_test_com11,pred2)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)


# ### SVM With SMOTE

# In[371]:


from sklearn.metrics import confusion_matrix, classification_report, f1_score
from imblearn.over_sampling import SMOTE 
from sklearn.svm import SVC #Support vector classifier
from sklearn.metrics import roc_auc_score, roc_curve
from confusionMatrix import plotConfusionMatrix
svm = SVC() #SVC(kernel='linear') 

clf = svm.fit(X_sm11, y_sm11.ravel()) 
score1 = svm.score(X_val11, y_val11)
score2 = svm.score(X_test_com11, y_test_com11)
pred1 = svm.predict(X_val11) 
pred2 = svm.predict(X_test_com11)  
# print classification report 
print('With SMOTE:'.upper())
print('Validation accuracy: ', score1)
print('test accuracy: ', score2)
print('*'*80)
print(confusion_matrix(y_test_com11, pred2))
print('*'*80)
print('F1 score val:\n', classification_report(y_val11, pred1)) 
print('*'*80)
#print('\n')
print('F1 score test:\n', classification_report(y_test_com11, pred2)) 

print('*'*80)


# In[372]:


# ROC Curve
svm_p = SVC(probability = True) # for probability
clf_p = svm_p.fit(X_sm11, y_sm11.ravel()) # for probability
pred_proba = svm_p.predict_proba(X_test_com11)[::,1]
fpr, tpr, _ = roc_curve(y_test_com11,  pred_proba)
auc = roc_auc_score(y_test_com11, pred_proba)
plt.plot(fpr,tpr,label="SVM With SMOTE ="+str(np.round(auc,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)
cnf_mat = confusion_matrix(y_test_com11,pred2)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)


# ### SVM with SMOTE on Test

# In[373]:



from sklearn.metrics import confusion_matrix, classification_report, f1_score
from imblearn.over_sampling import SMOTE 
from sklearn.svm import SVC #Support vector classifier
from sklearn.metrics import roc_auc_score, roc_curve
from confusionMatrix import plotConfusionMatrix
svm = SVC() #SVC(kernel='linear') 

clf = svm.fit(X_sm11, y_sm11.ravel()) 
score1 = svm.score(X_val11, y_val11)
score2 = svm.score(X_smt11, y_smt11)
pred1 = svm.predict(X_val11) 
pred2 = svm.predict(X_smt11)  
# print classification report 
print('With SMOTE:'.upper())
print('Validation accuracy: ', score1)
print('test accuracy: ', score2)
print('*'*80)
print(confusion_matrix(y_smt11, pred2))
print('*'*80)
print('F1 score val:\n', classification_report(y_val11, pred1)) 
print('*'*80)
#print('\n')
print('F1 score test:\n', classification_report(y_smt11, pred2)) 

print('*'*80)


# In[374]:


# ROC Curve
svm_p = SVC(probability = True) # for probability
clf_p = svm_p.fit(X_sm11, y_sm11.ravel()) # for probability
pred_proba = svm_p.predict_proba(X_smt11)[::,1]
fpr, tpr, _ = roc_curve(y_smt11,  pred_proba)
auc = roc_auc_score(y_smt11, pred_proba)
plt.plot(fpr,tpr,label="SVM With SMOTE ="+str(np.round(auc,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)
cnf_mat = confusion_matrix(y_smt11,pred2)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)


# ### SVM With UnderSampling on Train data

# In[375]:


from sklearn.metrics import confusion_matrix, classification_report, f1_score
from imblearn.over_sampling import SMOTE 
from sklearn.svm import SVC #Support vector classifier
from sklearn.metrics import roc_auc_score, roc_curve
from confusionMatrix import plotConfusionMatrix
svm = SVC() #SVC(kernel='linear') 

clf = svm.fit(X_sm12, y_sm12.ravel()) 
score1 = svm.score(X_val11, y_val11)
score2 = svm.score(X_test_com11, y_test_com11)
pred1 = svm.predict(X_val11) 
pred2 = svm.predict(X_test_com11)  
# print classification report 
print('With UnderSampling:'.upper())
print('Validation accuracy: ', score1)
print('test accuracy: ', score2)
print('*'*80)
print(confusion_matrix(y_test_com11, pred2))
print('*'*80)
print('F1 score val:\n', classification_report(y_val11, pred1)) 
print('*'*80)
#print('\n')
print('F1 score test:\n', classification_report(y_test_com11, pred2)) 

print('*'*80)


# In[376]:


# ROC Curve
svm_p = SVC(probability = True) # for probability
clf_p = svm_p.fit(X_sm11, y_sm11.ravel()) # for probability
pred_proba = svm_p.predict_proba(X_test_com11)[::,1]
fpr, tpr, _ = roc_curve(y_test_com11,  pred_proba)
auc = roc_auc_score(y_test_com11, pred_proba)
plt.plot(fpr,tpr,label="SVM With UnderSampling ="+str(np.round(auc,3)))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('*'*80)
cnf_mat = confusion_matrix(y_test_com11,pred2)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)


# In[ ]:




