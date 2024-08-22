#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


colnames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

#Load the file from local directory using pd.read_csv which is a special form of read_table
#while reading the data, supply the "colnames" list

pima_df = pd.read_csv("pima-indians-diabetes.data", names= colnames)


# In[3]:


pima_df.head(50)


# In[4]:


sns.countplot(x='class', data=pima_df)
plt.title('Count Plot of Category Column')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()


# In[5]:


# Let us look at the target column which is 'class' to understand how the data is distributed amongst the various values
pima_df.groupby(["class"]).count()


# In[6]:


pima_f = pima_df. isnull(). sum() * 100 / len(pima_df)
pima_f


# In[7]:


# Save the semi_clean data
pima_df.to_csv('clean_data.csv', index=False)


# In[ ]:




