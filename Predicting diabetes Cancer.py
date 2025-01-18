#!/usr/bin/env python
# coding: utf-8

# In[69]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


from sklearn.datasets import load_breast_cancer


# In[18]:


ld=load_breast_cancer()


# In[22]:


print(ld["DESCR"])


# In[40]:


df=pd.DataFrame(data=ld["data"],columns=ld["feature_names"])


# In[44]:


df["cancer"]=ld["target"]


# In[46]:


df


# In[48]:


df.info()


# In[50]:


df.describe()


# In[54]:


sns.pairplot(df,hue="cancer")


# In[ ]:


# 5 5 charta banaa
sns.pairplot(df[df.columns[:5]])
sns.pairplot(df[df.columns[5:10]])
sns.pairplot(df[df.columns[10:15]])


# In[71]:


for i in df.columns:
    print(i)
    sns.boxplot(data=df,x=i)
    plt.show()


# In[77]:


for i in df.columns:
    print(i)
    sns.histplot(data=df,x=i)
    plt.show()


# In[79]:


# ML start
X=df.drop("cancer",axis=1)
y=df["cancer"]


# In[81]:


from sklearn.model_selection import train_test_split


# In[ ]:


train_test_split


# In[83]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[85]:


from sklearn.linear_model import LogisticRegression


# In[91]:


lr=LogisticRegression()


# In[93]:


lr.fit(X_train, y_train)


# In[97]:


log_pred=lr.predict(X_test)


# In[102]:


from sklearn.metrics import confusion_matrix,classification_report


# In[104]:


confusion_matrix(y_test,log_pred)


# In[106]:


print(classification_report(y_test,log_pred))


# In[108]:


log_train_pred=log_pred=lr.predict(X_train)


# In[110]:


print(classification_report(y_train,log_train_pred))


# In[ ]:


# overfiting tarining high accuracy testing Low accuracy
# overfiting


# In[132]:


# Greed search
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

for i in [StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer]:
    print(i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
    ss=i()
    X_train=ss.fit_transform(X_train)
    X_test=ss.fit_transform(X_test)

    lr=LogisticRegression()

    lr.fit(X_train, y_train)

    log_train_pred=lr.predict(X_train)

    print("Training Recall",recall_score(y_train,log_train_pred))

    log_pred=lr.predict(X_test)

    print("Testing Recall",recall_score(y_test,log_pred))
    print("*"*30)


# In[142]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)

lr=LogisticRegression()

lr.fit(X_train, y_train)

log_train_pred=lr.predict(X_train)

print("Training Recall",recall_score(y_train,log_train_pred))

log_pred=lr.predict(X_test)

print("Testing Recall",recall_score(y_test,log_pred))
print("*"*30)




# In[146]:


from sklearn.neighbors import KNeighborsClassifier


# In[148]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)


# In[180]:


for i in [StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer]:
    print(i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
    ss=i()
    X_train=ss.fit_transform(X_train)
    X_test=ss.fit_transform(X_test)



    err=[]

    for j in range(2,31):

        knn=KNeighborsClassifier(n_neighbors=j)
        knn.fit(X_train, y_train)
    
        knn_train_pred=knn.predict(X_train)
    
        # print("Training Recall"recall_score(y_train,knn_train_pred))
    
        knn_pred=knn.predict(X_test)
        err.append(1-recall_score(y_test,knn_pred))

    plt.plot(range(2,31),err)
    plt.show()


# In[164]:


knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

knn_train_pred=knn.predict(X_train)

print("Training Recall",recall_score(y_train,knn_train_pred))

knn_pred=knn.predict(X_test)

print("Testing Recall",recall_score(y_test,knn_pred))


# In[ ]:




