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


df_house = pd.read_csv(r'E:\Dataset\Review_4_Ajarn\house_data_V.3_setTarget.txt', delimiter='\t')
df_land = pd.read_csv(r'E:\Dataset\Review_4_Ajarn\land_data_V.3_withtarget.txt', delimiter='\t')


# ### Clean Data --House_Dataset

# In[3]:


df_house


# #### Drop Columns with Missing Values

# In[4]:


df_house_droped = df_house.dropna(axis='columns')
df_house_droped


# #### Change String to Numeric Values

# In[5]:


from sklearn.preprocessing import LabelEncoder

# encoding string values into numeric values
le_propTy = LabelEncoder()
le_asseSt = LabelEncoder()
le_userTy = LabelEncoder()
le_homeCon = LabelEncoder()
le_roadTy = LabelEncoder()


# In[6]:


# create new columns containing numeric code of former column
df_house_droped['PropertyType_n'] = le_propTy.fit_transform(df_house_droped['PropertyType'])
df_house_droped['AsseStatus_n'] = le_asseSt.fit_transform(df_house_droped['AsseStatus'])
df_house_droped['UserType_n'] = le_userTy.fit_transform(df_house_droped['UserType'])
df_house_droped['HomeCondition_n'] = le_homeCon.fit_transform(df_house_droped['HomeCondition'])
df_house_droped['RoadType_n'] = le_roadTy.fit_transform(df_house_droped['RoadType'])
df_house_droped


# ### Select Columns that are used as Variables in Model

# In[7]:


df_house_select = df_house_droped[['PropertyType_n', 'SellPrice', 'CostestimateB','MarketPrice', 'HouseArea', 'Floor', 'HomeCondition_n', 'BuildingAge','RoadType_n','AsseStatus_n', 'UserType_n']]
df_house_select


# #### Export Cleaned Dataset

# In[8]:


df_export = df_house_select.to_csv(r'E:\Dataset\df_house_cleaned_export.txt', index=False)


# #### Prepare data & target value for Model training 

# In[9]:


# seperate train data and target data
df_train_house = df_house_select.drop('UserType_n', axis='columns')
target_house = df_house_select['UserType_n']


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


# encoding string values into numeric values
le_colorTy = LabelEncoder()
le_asseSt = LabelEncoder()
le_userTy = LabelEncoder()
le_raodTy = LabelEncoder()
le_groundLev = LabelEncoder()


# In[15]:


# create new columns containing numeric code of former column
df_land_droped['ColorType_n'] = le_colorTy.fit_transform(df_land_droped['ColorType'])
df_land_droped['AsseStatus_n'] = le_asseSt.fit_transform(df_land_droped['AsseStatus'])
df_land_droped['UserType_n'] = le_userTy.fit_transform(df_land_droped['UserType'])
df_land_droped['RoadType_n'] = le_roadTy.fit_transform(df_land_droped['RoadType'])
df_land_droped['GroundLevel_n'] = le_groundLev.fit_transform(df_land_droped['GroundLevel'])
df_land_droped


# ### Select Columns that are used as Variables in Model

# In[16]:


df_land_select = df_land_droped[['ColorType_n', 'CostestimateB', 'SellPrice', 'MarketPrice',  'RoadType_n', 'AsseStatus_n', 'UserType_n']]
df_land_select


# #### Export Cleaned Dataset

# In[17]:


df_export = df_land_select.to_csv(r'E:\Dataset\df_land_cleaned_export.txt', index=False)


# ### Prepare data & target value for Model training 

# In[18]:


# seperate train data and target data
df_train_land = df_land_select.drop('UserType_n', axis='columns')
target_land = df_land_select['UserType_n']


# In[19]:


df_train_land


# In[20]:


target_land


# ### Decision Tree Clasifier (House)

# In[21]:


from sklearn.tree import DecisionTreeClassifier


# #### Data Slicing
# ##### Split Dataset into Train and Test

# In[22]:


from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
XH = df_train_house
YH = target_house
XH_train, XH_test, YH_train, YH_test = train_test_split(XH, YH, test_size=0.3, random_state=100)


# In[23]:


XH_train.shape, YH_train.shape


# In[24]:


XH_test.shape, YH_test.shape


# #### Train Dataset
# ##### Gini Index

# In[25]:


# Classifier Object
clf_giniHouse = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5) 


# In[26]:


# Perform Training 
clf_giniHouse.fit(XH_train, YH_train) 
clf_giniHouse


# ##### Entropy

# In[27]:


# Decision tree with entropy 
clf_entropyHouse = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5) 


# In[28]:


# Performing training 
clf_entropyHouse.fit(XH_train, YH_train) 
clf_entropyHouse 


# #### Prediction & Accuracy
# ##### Prediction using Gini or Entropy

# In[29]:


# Predicton on test with giniIndex 
YH_pred = clf_giniHouse.predict(XH_test) 
print("Predicted values:") 
YH_pred


# In[30]:


# Predicton on test with Entropy 
YH_pred = clf_entropyHouse.predict(XH_test) 
print("Predicted values:") 
YH_pred


# ##### Calculate Accuracy

# In[31]:


from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 


# In[32]:


print("Confusion Matrix: ", confusion_matrix(YH_test, YH_pred)) 
print ("Accuracy : ", accuracy_score(YH_test, YH_pred)*100) 
print("Report : ", classification_report(YH_test, YH_pred)) 


# ### Decision Tree Clasifier (Land)

# #### Data Slicing
# ##### Split Dataset int Train and Test

# In[33]:


from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
XL = df_train_land
YL = target_land
XL_train, XL_test, YL_train, YL_test = train_test_split(XL, YL, test_size=0.3, random_state=100)


# In[34]:


XL_train.shape, YL_train.shape


# In[35]:


XL_test.shape, YL_test.shape


# #### Train Dataset
# ##### Gini Index

# In[36]:


# Decision Tree with Gini Index
clf_giniLand = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5) 


# In[37]:


# Perform Training 
clf_giniLand.fit(XL_train, YL_train) 
clf_giniLand


# ##### Entropy

# In[38]:


# Decision Tree with Entropy 
clf_entropyLand = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5) 


# In[39]:


# Performing training 
clf_entropyLand.fit(XL_train, YL_train) 
clf_entropyLand


# #### Prediction & Accuracy
# ##### Prediction using Gini or Entropy

# In[40]:


# Predicton on test with giniIndex 
YL_pred = clf_giniLand.predict(XL_test) 
print("Predicted values:") 
YL_pred


# In[41]:


# Predicton on test with Entropy 
YL_pred = clf_entropyLand.predict(XL_test) 
print("Predicted values:") 
YL_pred


# ##### Calculate Accuracy

# In[42]:


print("Confusion Matrix: ", confusion_matrix(YL_test, YL_pred)) 
print ("Accuracy : ", accuracy_score(YL_test, YL_pred)*100) 
print("Report : ", classification_report(YL_test, YL_pred)) 

