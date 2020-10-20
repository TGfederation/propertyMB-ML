#!/usr/bin/env python
# coding: utf-8

# # Environment conda3--python3
# ## Coding UTF-8
# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import missingno as msno

# show library versions for documentation reference
import sys
print("Python: {}".format(sys.version))
print("pandas: {}".format(pd.__version__))
print("numpy: {}".format(np.__version__))
print("seaborn: {}".format(sns.__version__))
print("missingno: {}".format(msno.__version__))


# ### Load Land Dataset

# In[2]:


df_land_new = pd.read_excel(r'C:\Users\OS\Desktop\Model Improvement Dataset\Land-New Dataset 08-2020.xlsx')
df_land_current = pd.read_excel(r'C:\Users\OS\Desktop\Model Improvement Dataset\Land-Current Dataset 09-2020.xlsx')


# ### Show Data --Land_Dataset

# In[3]:


# show example data from loaded file
df_land_new


# In[4]:


df_land_current


# #### Visualisatuion assisting Analysis of Data

# In[5]:


#frequency of label in UserType from new Dataset
df_land_new['UserType'].value_counts()


# In[6]:


#frequency of label in UserType from old Dataset
fr_current_df = df_land_current['UserType'].value_counts()
fr_current_df


# In[7]:


#plotting categorical variables for visualisation
fr_current_df.plot(kind='bar')


# In[8]:


#The next single-line code will visualize the location of missing values.
sns.heatmap(df_land_current.isnull(), cbar=False)
"""We need to know if the occurrence of missing values are sparsely located or located as a big chunk. This heatmap visualization immediately tells us such tendency. Also, if more than 2 columns have correlation in missing value locations, such correlation will be visualized."""


# In[9]:


# Visualize missing values as a Matrix
msno.matrix(df_land_current, labels=True)
"""In addition to the heatmap, there is a bar on the right side of this diagram. This is a line plot for each row's data completeness.Also, missingno.heatmap visualizes the correlation matrix about the locations of missing values in columns."""


# In[10]:


# Visualiz the missing values as a Bar Chart
msno.bar(df_land_current, labels=True, sort="ascending")
"""A simple representation of nullity by column"""


# In[11]:


# Visualize the correlation between the number of  missing values in different columns as a Heatmap
msno.heatmap(df_land_current)
"""The bar chart of the number of missing values in each column and the dendrogram generated from the correlation of missing value locations. The missingno correlation heatmap measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another."""


# In[12]:


# Nullity correlation ranges from -1 (if one variable appears the other definitely does not) to 0 
#(variables appearing or not appearing have no effect on one another) to 1 
#(if one variable appears the other definitely also does).

# Entries marked <1 or >-1 are have a correlation that is close to being exactingly negative or positive, 
#but is still not quite perfectly so. This points to a small number of records in the dataset which are erroneous. 
#For example, in this dataset the correlation between WxD and ContactSo 3 is -0.9, indicating that, 
#contrary to our expectation, there are a few records which have one or the other, but not both. 
#These cases will require special attention.

# The heatmap works great for picking out data completeness relationships between variable pairs, 
#but its explanatory power is limited when it comes to larger relationships and it has no particular support 
#for extremely large datasets.


# #### Deal with missing Values

# In[13]:


# Confirm the number of missing values in each column.
df_land_current.info()


# In[14]:


# Checking for missing values in each column
df_land_current.isnull().sum()


# In[15]:


# Replace RoadType index that contain missing values wth NaN, in order to be able to change string to numeric
# In this case 1 row was replaced in RoadType column.
df_land_current = df_land_current.replace(np.nan).dropna(subset=['RoadType'])
df_land_current


# In[16]:


# Check if the missing values were replaced.
df_land_current.isnull().sum()


# #### Change String to Numeric Value

# In[17]:


from sklearn.preprocessing import LabelEncoder

# make a copy of the datframe
df_land_updated = df_land_current.copy(deep=True)

# encoding string values into numeric values
le_colorTy = LabelEncoder()
le_asseSt = LabelEncoder()
le_userTy = LabelEncoder()
le_roadTy = LabelEncoder()
le_groundLev = LabelEncoder()

df_land_updated


# In[18]:


# create new columns containing numeric code of former column
df_land_current['ColorType_n'] = le_colorTy.fit_transform(df_land_current['ColorType'])
df_land_current['AsseStatus_n'] = le_asseSt.fit_transform(df_land_current['AsseStatus'])
df_land_current['UserType_n'] = le_userTy.fit_transform(df_land_current['UserType'])
df_land_current['RoadType_n'] = le_roadTy.fit_transform(df_land_current['RoadType'])
df_land_current['GroundLevel_n'] = le_groundLev.fit_transform(df_land_current['GroundLevel'])
df_land_current


# #### Prepare dataset for model training

# In[19]:


df_land_select = df_land_current[['ColorType_n', 'CostestimateB', 'SellPrice', 'MarketPrice',  'RoadType_n', 'AsseStatus_n', 'UserType_n']]
df_land_select


# #### Export Cleaned Dataset

# In[20]:


# model training dataset
df_land_select.to_csv(r'C:\Users\OS\Desktop\Model Improvement Dataset\df_cleaned_land_model_001-09-2020.txt', index=False)
df_land_select.to_csv(r'C:\Users\OS\Desktop\Model Improvement Dataset\df_cleaned_land_model_001-09-2020.csv', index=False)

# entire dataset
df_land_updated.to_csv(r'C:\Users\OS\Desktop\Model Improvement Dataset\df_cleaned_land_001-09-2020.txt', index=False)

