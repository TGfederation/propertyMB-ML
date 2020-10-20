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


# ### Load House Dataset

# In[2]:


df_house_new = pd.read_excel(r'C:\Users\OS\Desktop\Model Improvement Dataset\House-New Dataset 08-2020.xlsx')
df_house_current = pd.read_excel(r'C:\Users\OS\Desktop\Model Improvement Dataset\House-Current Dataset 08-2020.xlsx')


# ### Show Data --House_Dataset

# In[3]:


# show example data from loaded file
df_house_new


# In[4]:


df_house_current


# #### Visualisatuion assisting Analysis of Data

# In[5]:


#frequency of labele in UserType from new Dataset
fr_new_df = df_house_new['UserType'].value_counts()
fr_new_df


# In[6]:


#frequency of labele in UserType from old Dataset
fr_current_df = df_house_current['UserType'].value_counts()
fr_current_df


# In[7]:


#plotting categorical variables for visualisation
fr_new_df.plot(kind='bar')


# In[8]:


fr_current_df.plot(kind='bar')


# In[9]:


# Visualize the location of missing values.
sns.heatmap(df_house_current.isnull(), cbar=False)
"""We need to know if the occurrence of missing values are sparsely located or located as a big chunk. This heatmap visualization immediately tells us such tendency. Also, if more than 2 columns have correlation in missing value locations, such correlation will be visualized."""


# In[10]:


# Visualize missing values as a Matrix
msno.matrix(df_house_current, labels=True, fontsize=10)
"""In addition to the heatmap, there is a bar on the right side of this diagram. This is a line plot for each row's data completeness.Also, missingno.heatmap visualizes the correlation matrix about the locations of missing values in columns."""


# In[11]:


# Visualiz the missing values as a Bar Chart
msno.bar(df_house_current, labels=True, fontsize=10, sort="ascending")
"""A simple representation of nullity by column"""


# In[12]:


# Visualize the correlation between the number of missing values in different columns as a Heatmap
msno.heatmap(df_house_current)
"""The bar chart of the number of missing values in each column and the dendrogram generated from the correlation of missing value locations. The missingno correlation heatmap measures nullity correlation: how strongly the presence or absence of one variable affects the presence of another."""


# In[13]:


# Nullity correlation ranges from -1 (if one variable appears the other definitely does not) to 0 
#(variables appearing or not appearing have no effect on one another) to 1 (if one variable appears 
#the other definitely also does).

# Entries marked <1 or >-1 have a correlation that is close to being exactingly negative or positive, 
#but is still not quite perfectly so. This points to a small number of records in the dataset which are erroneous. 
#For example, in this dataset the correlation between LandAge and Letc is >-1, indicating that, 
#contrary to our expectation, there are a few records which have one or the other, but not both. 
#These cases will require special attention.

# The heatmap works great for picking out data completeness relationships between variable pairs, 
#but its explanatory power is limited when it comes to larger relationships and it has no particular support
#for extremely large datasets.


# #### Deal with missing Values

# In[14]:


# Confirm the number of missing values in each column.
df_house_current.info()


# In[15]:


# Replace RoadType index that contain missing values wth NaN, in order to be able to change string to numeric
# In this case 1 row was replaced in RoadType column.
df_house_current = df_house_current.replace(np.nan).dropna(subset=['PPStatus'])
df_house_current


# In[16]:


# Check if the missing values were replaced.
df_house_current.info()


# #### Change String to Numeric Values

# In[17]:


from sklearn.preprocessing import LabelEncoder

# make a copy of the datframe
df_house_updated = df_house_current.copy(deep=True)

# encoding string values into numeric values
le_propTy = LabelEncoder()
le_asseSt = LabelEncoder()
le_userTy = LabelEncoder()
le_homeCon = LabelEncoder()
le_roadTy = LabelEncoder()

df_house_updated


# In[18]:


# create new columns containing numeric code of former column
df_house_current['PropertyType_n'] = le_propTy.fit_transform(df_house_current['PropertyType'])
df_house_current['AsseStatus_n'] = le_asseSt.fit_transform(df_house_current['AsseStatus'])
df_house_current['UserType_n'] = le_userTy.fit_transform(df_house_current['UserType'])
df_house_current['HomeCondition_n'] = le_homeCon.fit_transform(df_house_current['HomeCondition'])
df_house_current['RoadType_n'] = le_roadTy.fit_transform(df_house_current['RoadType'])
df_house_current


# In[19]:


df_house_select = df_house_current[['PropertyType_n', 'SellPrice', 'CostestimateB','MarketPrice', 
                                   'HouseArea', 'Floor', 'HomeCondition_n', 'BuildingAge','RoadType_n',
                                   'AsseStatus_n', 'UserType_n']]
df_house_select


# #### Export Cleaned Dataset

# In[20]:


# model training dataset as text and csv
df_house_select.to_csv(r'C:\Users\OS\Desktop\Model Improvement Dataset\df_cleaned_house_model_001-09-2020.txt', index=False)
df_house_select.to_csv(r'C:\Users\OS\Desktop\Model Improvement Dataset\df_cleaned_house_model_001-09-2020.csv', index=False)

# entire dataset as text (encoding error with csv)
df_house_updated.to_csv(r'C:\Users\OS\Desktop\Model Improvement Dataset\df_cleaned_house_001-09-2020.txt', index=False)

