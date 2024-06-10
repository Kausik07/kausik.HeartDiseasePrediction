#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing
'exec(% matplotlib inline)'
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns


# In[3]:


import pandas as pd

# Load the dataset
disease_df = pd.read_csv(r"C:\Users\Thamaiyanthi\Documents\Arathi-Projects\ML Project\Heart_Disease_Prediction.csv")

# Drop the 'education' column if it exists
if 'education' in disease_df.columns:
    disease_df.drop(['education'], inplace=True, axis=1)

# Rename the 'male' column to 'Sex_male'
disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)

# Print the first few rows of the modified dataset
print(disease_df.head())



# In[5]:


# removing NaN / NULL values
disease_df.dropna(axis=0, inplace=True)
print(disease_df.head(), disease_df.shape)

# Check if 'TenYearCHD' column exists before trying to access it
if 'TenYearCHD' in disease_df.columns:
    print(disease_df['TenYearCHD'].value_counts())
else:
    print("Column 'TenYearCHD' does not exist in the DataFrame.")



# In[6]:


# Define X and y based on your dataset column names
X = np.asarray(disease_df[['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium']])
y = np.asarray(disease_df['Heart Disease'])

# Normalization of the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

# Train-and-Test-Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size = 0.3, random_state = 4)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)



# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
sns.countplot(x='Heart Disease', data=disease_df, palette="BuGn_r")
plt.show()


# In[10]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)




# In[11]:


# Evaluation and accuracy
from sklearn.metrics import accuracy_score
print('Accuracy of the model is =', 
      accuracy_score(y_test, y_pred))


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming cm is your confusion matrix
conf_matrix = pd.DataFrame(data=cm, 
                           columns=['Predicted:0', 'Predicted:1'], 
                           index=['Actual:0', 'Actual:1'])

plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Greens")
plt.show()
print('The details for confusion matrix is =')



# In[ ]:




