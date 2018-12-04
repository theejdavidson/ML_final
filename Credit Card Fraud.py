
# coding: utf-8

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np


df = pd.read_csv("creditcard.csv", header = 0)


# In[29]:


from sklearn.model_selection import train_test_split

X= df.iloc[1:10000, [1, 2, 3, 4, 5]].values
y = df.iloc[1:10000, [30]].values
y = np.where(y== 0, -1, 1)
y = y.reshape((9999, ))
#print(X.shape)
#print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
X_test, X_dev, y_test, y_dev= train_test_split(X_test, y_test, test_size=0.5, random_state=1)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
#y_test = y_test.reshape((1500, ))
print(y_test.shape)


# In[31]:


svm = svm.SVC(kernel= 'linear', C = 1, gamma = 1)
svm.fit(X_test, y_test)

