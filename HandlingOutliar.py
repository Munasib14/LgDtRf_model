#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# In[2]:


pima_df = pd.read_csv("clean_data.csv")
pima_df


# In[3]:


# Function to compute IQR and outliers
def calculate_outliers(df, column):
    q1 = pima_df.quantile(0.25)
    q3 = pima_df.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = pima_df[(pima_df < lower_bound) | (pima_df > upper_bound)]
    return outliers, lower_bound, upper_bound, q1, q3, iqr


# In[4]:


# Calculate outliers and plot boxplots for each column
# Select numeric columns
numeric_columns = pima_df.select_dtypes(include=[np.number]).columns

for column in numeric_columns:
    outliers, lower_bound, upper_bound, q1, q3, iqr = calculate_outliers(pima_df, column)


# In[5]:


print(f"Outliers in column '{column}':")
print(outliers)

# Plot boxplot
pima_df.boxplot(column=column)
plt.title(f'Boxplot of {column}')
plt.show()


# In[6]:


plt.rcParams['figure.figsize'] = [9,5]
pima_df.hist()
plt.tight_layout()
plt.show()


# In[7]:


pima_df.to_csv('cleaned_data.csv', index=False)


# In[ ]:




