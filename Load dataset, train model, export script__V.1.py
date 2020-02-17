#!/usr/bin/env python
# coding: utf-8

# # Environment conda3--python3
# ## Coding UTF-8
# ### Import Libraries

# In[1]:


# turn off feature warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# show library versions for documentation reference
import sys
print("Python: {}".format(sys.version))
print("pandas: {}".format(pd.__version__))
print("numpy: {}".format(np.__version__))


# ### Load Dataset

# In[2]:


df_house = pd.read_csv(r'C:\Users\User\Desktop\Dataset_v02\house_with_target_002.txt', delimiter='\t')
df_land  = pd.read_csv(r'C:\Users\User\Desktop\Dataset_v02\lands_dataset.txt', delimiter='\t')


# ### Clean Data --House_Dataset

# In[3]:


# show data from loaded file
df_house


# #### Drop Columns with Missing Values

# In[4]:


df_house_droped = df_house.dropna(axis='columns')
df_house_droped


# #### Change String to Numeric Values

# In[5]:


from sklearn.preprocessing import LabelEncoder

# encoding string values into numeric values
le_idProp = LabelEncoder()
le_propTy = LabelEncoder()
le_asseSt = LabelEncoder()
le_userTy = LabelEncoder()
le_houseAr = LabelEncoder()
le_floor = LabelEncoder()


# In[6]:


# create new columns containing numeric code of former column
df_house_droped['ID_Property_n'] = le_idProp.fit_transform(df_house_droped['ID_Property'])
df_house_droped['PropertyType_n'] = le_propTy.fit_transform(df_house_droped['PropertyType'])
df_house_droped['AsseStatus_n'] = le_asseSt.fit_transform(df_house_droped['AsseStatus'])
df_house_droped['UserType_n'] = le_userTy.fit_transform(df_house_droped['UserType'])
df_house_droped['HouseArea_n'] = le_houseAr.fit_transform(df_house_droped['HouseArea'])
df_house_droped['Floor_n'] = le_floor.fit_transform(df_house_droped['Floor'])
df_house_droped


# In[7]:


#drop columns with old string values
df_house_cleaned = df_house_droped.drop(['ID_Property', 
                        'PropertyType',
                        'AsseStatus',
                        'UserType',
                        'HouseArea',
                        'Floor',
                        'Owner'
                       ], axis='columns')
df_house_cleaned


# #### Export Cleaned Dataset

# In[8]:


df_export = df_house_cleaned.to_csv(r'C:\Users\User\Desktop\Dataset_v02\df_house_cleaned.txt', index=False)


# #### Prepare data & target value for Model training 

# In[9]:


# seperate train data and target data
df_train_house = df_house_cleaned.drop('UserType_n', axis='columns')
target_house = df_house_cleaned['UserType_n']


# In[10]:


df_train_house


# In[11]:


target_house


# ### Clean Data --Land_Dataset

# In[12]:


# show example data from loaded file
df_land


# #### Drop Columns with missing Values

# In[13]:


df_land_droped = df_land.dropna(axis='columns')
df_land_droped


# #### Change String to Numeric Value

# In[14]:


from sklearn.preprocessing import LabelEncoder

# encoding string values into numeric values
le_idLands = LabelEncoder()
le_colorTy = LabelEncoder()
le_asseSt = LabelEncoder()
le_LAmp = LabelEncoder()
le_LDis = LabelEncoder()
le_userTy = LabelEncoder()


# In[15]:


# create new columns containing numeric code of former column
df_land_droped['ID_Lands_n'] = le_idProp.fit_transform(df_land_droped['ID_Lands'])
df_land_droped['ColorType_n'] = le_colorTy.fit_transform(df_land_droped['ColorType'])
df_land_droped['AsseStatus_n'] = le_asseSt.fit_transform(df_land_droped['AsseStatus'])
df_land_droped['LAmphur_n'] = le_LAmp.fit_transform(df_land_droped['LAmphur'])
df_land_droped['LDistrict_n'] = le_LDis.fit_transform(df_land_droped['LDistrict'])
df_land_droped['UserType_n'] = le_userTy.fit_transform(df_land_droped['UserType'])
df_land_droped


# In[16]:


#drop columns with old string values
df_land_cleaned = df_land_droped.drop(['ID_Lands', 
                        'ColorType',
                        'AsseStatus',
                        'LAmphur',
                        'LDistrict',
                        'UserType',
                        'Owner'
                       ], axis='columns')
df_land_cleaned


# #### Export CLeaned Dataset

# In[17]:


df_export = df_land_cleaned.to_csv(r'C:\Users\User\Desktop\Dataset_v02\df_land_cleaned.txt', index=False)


# ### Prepare data & target value for Model training 

# In[18]:


# seperate train data and target data
df_train_land = df_land_cleaned.drop('UserType_n', axis='columns')
target_land = df_land_cleaned['UserType_n']


# In[19]:


df_train_land


# In[20]:


target_land


# ### Decision Tree Clasifier

# In[21]:


from sklearn import tree


# #### For House

# In[22]:


model_house = tree.DecisionTreeClassifier()

model_house.fit(df_train_house, target_house)


# In[23]:


model_house.score(df_train_house, target_house)


# #### For Land

# In[24]:


model_land = tree.DecisionTreeClassifier()

model_land.fit(df_train_land, target_land)


# In[25]:


model_land.score(df_train_land, target_land)


# ## Save ML Model

# In[26]:


import pickle

with open(r'C:\Users\User\Desktop\ML_Model\Model_V.1\Model_Pickle_House', 'wb') as mlA:
    pickle.dump(model_house,mlA)
with open(r'C:\Users\User\Desktop\ML_Model\Model_V.1\Model_Pickle_Land', 'wb') as mlB:
    pickle.dump(model_land,mlB)


# In[27]:


with open(r'C:\Users\User\Desktop\ML_Model\Model_V.1\Model_Pickle_House', 'rb') as mlA:
    mp_House = pickle.load(mlA)


# In[28]:


with open(r'C:\Users\User\Desktop\ML_Model\Model_V.1\Model_Pickle_Land', 'rb') as mlB:
    mp_Land  = pickle.load(mlB)


# In[29]:


# mp stands for model pickle
mp_Land.predict


# In[ ]:




