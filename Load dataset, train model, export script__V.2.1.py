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
# ##### Gini Index = a metric to measure how often a randomly chosen element would be incorrectly identified. It means an attribute with lower gini index should be preferred.

# In[25]:


# Classifier Object
clf_giniHouse = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5) 


# In[26]:


# Perform Training 
clf_giniHouse.fit(XH_train, YH_train) 
clf_giniHouse


# ##### Entropy = the measure of uncertainty of a random variable, it characterizes the impurity of an arbitrary collection of examples. The higher the entropy the more the information content.

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
YHgini_pred = clf_giniHouse.predict(XH_test) 
print("Predicted values:") 
YHgini_pred


# In[30]:


# Predicton on test with Entropy 
YHentropy_pred = clf_entropyHouse.predict(XH_test) 
print("Predicted values:") 
YHentropy_pred


# ##### Calculate Accuracy

# In[31]:


from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 


# ### Gini Accuracy

# In[32]:


print("Confusion Matrix: ", confusion_matrix(YH_test, YHgini_pred)) 
# Confusion Matrix = a summary of prediction results on a classification problem. The number of correct and incorrect predictions are summarized with count values and broken down by each class. 
#                           It shows the ways in which your classification model is confused when it makes predictions. It gives us insight not only into the errors being made by a classifier but more importantly the types of errors that are being made.
#                                                        Predicted: No    Predicted: Yes
#                               Actual: No         [      TN                  FP           ]
#                               Actual: Yes        [      FN                  TP           ]
print ("Accuracy : ", accuracy_score(YH_test, YHgini_pred)*100) 
# Accuracy = (TP + TN) / (TP + TN + FP + FN); True Positive (TP) : Observation is positive, and is predicted to be positive. 
# False Negative (FN) : Observation is positive, but is predicted negative. 
# True Negative (TN) : Observation is negative, and is predicted to be negative. 
# False Positive (FP) : Observation is negative, but is predicted positive.
print("Report : ", classification_report(YH_test, YHgini_pred)) 
# Precision = TP / (TP + TN) High Precision indicates an example labelled as positive is indeed positive (a small number of FP). Recall = TP / (TP + FN)  High Recall indicates the class is correctly recognized (a small number of FN). 
## f1-score = (2*Recall*Precision) / (Recall + Precision) Since we have two measures (Precision and Recall) it helps to have a measurement that represents both of them. 
##                We calculate an F-measure which uses Harmonic Mean in place of Arithmetic Mean as it punishes the extreme values more. The F-Measure will always be nearer to the smaller value of Precision or Recall.
## High recall, low precision: This means that most of the positive examples are correctly recognized (low FN) but there are a lot of false positives. 
## Low recall, high precision: This shows that we miss a lot of positive examples (high FN) but those we predict as positive are indeed positive (low FP) 
## Support = amount of elements/target (in this case the amount of UserType)


# ### Entropy Accuracy

# In[33]:


print("Confusion Matrix: ", confusion_matrix(YH_test, YHentropy_pred)) 
print ("Accuracy : ", accuracy_score(YH_test, YHentropy_pred)*100) 
print("Report : ", classification_report(YH_test, YHentropy_pred)) 


# ### Decision Tree Clasifier (Land)

# #### Data Slicing
# ##### Split Dataset int Train and Test

# In[34]:


from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
XL = df_train_land
YL = target_land
XL_train, XL_test, YL_train, YL_test = train_test_split(XL, YL, test_size=0.3, random_state=100)


# In[35]:


XL_train.shape, YL_train.shape


# In[36]:


XL_test.shape, YL_test.shape


# #### Train Dataset
# ##### Gini Index = a metric to measure how often a randomly chosen element would be incorrectly identified. It means an attribute with lower gini index should be preferred.

# In[37]:


# Decision Tree with Gini Index
clf_giniLand = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3, min_samples_leaf=5) 


# In[38]:


# Perform Training 
clf_giniLand.fit(XL_train, YL_train) 
clf_giniLand


# ##### Entropy = the measure of uncertainty of a random variable, it characterizes the impurity of an arbitrary collection of examples. The higher the entropy the more the information content.

# In[39]:


# Decision Tree with Entropy 
clf_entropyLand = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5) 


# In[40]:


# Performing training 
clf_entropyLand.fit(XL_train, YL_train) 
clf_entropyLand


# #### Prediction & Accuracy
# ##### Prediction using Gini or Entropy

# In[41]:


# Predicton on test with giniIndex 
YLgini_pred = clf_giniLand.predict(XL_test) 
print("Predicted values:") 
YLgini_pred


# In[42]:


# Predicton on test with Entropy 
YLentropy_pred = clf_entropyLand.predict(XL_test) 
print("Predicted values:") 
YLentropy_pred


# #### Calculate Accuracy

# ###### Confusion Matrix = a summary of prediction results on a classification problem. The number of correct and incorrect predictions are summarized with count values and broken down by each class. It shows the ways in which your classification model is confused when it makes predictions. It gives us insight not only into the errors being made by a classifier but more importantly the types of errors that are being made.
# ###### Accuracy = (TP + TN) / (TP + TN + FP + FN); True Positive (TP) : Observation is positive, and is predicted to be positive. False Negative (FN) : Observation is positive, but is predicted negative. True Negative (TN) : Observation is negative, and is predicted to be negative. False Positive (FP) : Observation is negative, but is predicted positive.
# ###### Precision = TP / (TP + TN) High Precision indicates an example labelled as positive is indeed positive (a small number of FP). Recall = TP / (TP + FN)  High Recall indicates the class is correctly recognized (a small number of FN). 
# ###### f1-score = (2*Recall*Precision) / (Recall + Precision) Since we have two measures (Precision and Recall) it helps to have a measurement that represents both of them. We calculate an F-measure which uses Harmonic Mean in place of Arithmetic Mean as it punishes the extreme values more. The F-Measure will always be nearer to the smaller value of Precision or Recall.
# ###### High recall, low precision: This means that most of the positive examples are correctly recognized (low FN) but there are a lot of false positives. 
# ###### Low recall, high precision: This shows that we miss a lot of positive examples (high FN) but those we predict as positive are indeed positive (low FP) 
# ###### Support = amount of elements/target (in this case the amount of UserType)

# ### Gini Accuracy

# In[43]:


print("Confusion Matrix: ", confusion_matrix(YL_test, YLgini_pred)) 
print ("Accuracy : ", accuracy_score(YL_test, YLgini_pred)*100) 
print("Report : ", classification_report(YL_test, YLgini_pred)) 


# ### Entropy Accuracy

# In[44]:


print("Confusion Matrix: ", confusion_matrix(YL_test, YLentropy_pred)) 
print ("Accuracy : ", accuracy_score(YL_test, YLentropy_pred)*100) 
print("Report : ", classification_report(YL_test, YLentropy_pred)) 


# #### Export Model with Pickle
# ##### House Model

# In[45]:


import pickle 


# In[46]:


with open(r'E:\Model\Model_Pickle_giniHouse_V01', 'wb') as mlA:
    pickle.dump(clf_giniHouse,mlA)
with open(r'E:\Model\Model_Pickle_entropyHouse_V01', 'wb') as mlA:
    pickle.dump(clf_entropyHouse,mlA)


# ##### Land Model

# In[47]:


with open(r'E:\Model\Model_Pickle_giniLand_V01', 'wb') as mlA:
    pickle.dump(clf_giniLand,mlA)
with open(r'E:\Model\Model_Pickle_entropyLand_V01', 'wb') as mlA:
    pickle.dump(clf_entropyLand,mlA)


# #### Load House Pickle Model

# In[48]:


with open(r'E:\Model\Model_Pickle_giniHouse_V01', 'rb') as mlA:
    modelPickle_giniHouse = pickle.load(mlA)


# In[49]:


modelPickle_giniHouse.predict


# In[50]:


with open(r'E:\Model\Model_Pickle_entropyHouse_V01', 'rb') as mlA:
    modelPickle_entropyHouse = pickle.load(mlA)


# In[51]:


modelPickle_entropyHouse.predict


# #### Load Land Pickle Model

# In[52]:


with open(r'E:\Model\Model_Pickle_giniLand_V01', 'rb') as mlA:
    modelPickle_giniLand = pickle.load(mlA)


# In[53]:


modelPickle_giniLand.predict


# In[54]:


with open(r'E:\Model\Model_Pickle_entropyLand_V01', 'rb') as mlA:
    modelPickle_entropyLand = pickle.load(mlA)


# In[55]:


modelPickle_entropyLand.predict


# In[61]:


#class distribution
print(df_house_droped.groupby('UserType').size())


# In[56]:


#class distribution
print(df_land_droped.groupby('UserType').size())


# In[59]:


# descriptions
print(df_house_select.describe())


# In[60]:


# descriptions
print(df_land_select.describe())


# In[ ]:




