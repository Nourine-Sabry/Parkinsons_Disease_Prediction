#!/usr/bin/env python
# coding: utf-8

# ### Nourine Sabry - 211000198
# # Parkinson's Disease Prediction Using Machine Learning #
# ***
# <center> <img src="Parkinsons-disease-3.jpg" width="500" height="300"> <center>
# <Center> Parkinson's disease (PD) is the second most common progressive neurodegenerative disease. PD symptoms include tremor, rigidity, cognitive impairment, and gastrointestinal issues. Diagnosing PD often relies on medical observations of motor symptoms, but in cases of early non-motor symptoms, subtle and mild symptoms might be overlooked. The aim of this project is to implement a machine learning model to try and predict whether a patient has PD or not based on speech features.</center>
# 
# ***
# **Link to dataset:** https://www.kaggle.com/datasets/dipayanbiswas/parkinsons-disease-speech-signal-features/data 

# ### Importing libraries

# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import math


# ### Import dataset

# In[58]:


data=pd.read_csv("pd_speech_features.csv")


# ### Data exploration & preprocessing

# In[59]:


data.info()


# In[60]:


data.describe()


# In[61]:


data.head()


# In[5]:


data.shape


# ### PD usually affects more males than females, let's see if the data agrees with this:

# In[6]:


heatmapdata = pd.crosstab(data['class'], data['gender'])
sns.heatmap(heatmapdata, yticklabels=['Healthy', 'PD'], xticklabels=['Female', 'Male'], annot=True, fmt='d')
plt.title('Distribution of males & females with & without PD')
plt.show()


# ### Let's see how correlated the features are:

# In[14]:


corr_matr = data.drop(columns=['id', 'gender']).corr(method='pearson')
plt.figure(figsize=(5,5))
sns.heatmap(corr_matr, cmap='coolwarm', square=True)
plt.title("Correlation heatmap on raw PD dataset")
plt.show()


# ### Uh oh, highly correlated features are not good! Let's remove them:

# **Q: What are highly correlated features? And what's so bad about them?**
# - **A:** Highly correlated features are variables that have a strong linear relationship with each other; most medical datasets tend to have highly correlated features. Highly correlated features are redundant and can increase the complexity of the model. Also, highly correlated features make it difficult to determine the importance of each feature and thus introduce ambiguity. Overall, it is crucial to remove highly correlated features in order to improve model performance and interpretability, as well as reduce the computational power needed, making it faster to train the model.
# - **Further reading:** https://medium.com/@sujathamudadla1213/why-we-have-to-remove-highly-correlated-features-in-machine-learning-9a8416286f18

# In[65]:


X = data.drop('class', axis=1)
X_norm = MinMaxScaler().fit_transform(X)
selector = SelectKBest(chi2, k=30)
selector.fit(X_norm, data['class'])
filtered_columns = selector.get_support()
filtered_data = X.loc[:, filtered_columns]
filtered_data['class'] = data['class']
data = filtered_data


# In[13]:


data.shape


# ### Good! We've reduced the number of feature to 30 instead of the overwhelming 755!

# ### Now let's see how balanced the data is:

# In[33]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
piedata = data['class'].value_counts()
ax1.pie(piedata.values, labels=piedata.index, autopct='%1.1f%%')
countimbalanced=sns.countplot(data=data, x='class', ax=ax2)
countimbalanced.set(ylabel='No. of Samples', xlabel='Have Parkinson?')
plt.tight_layout()
plt.show()


# ### Yikes! The data is highly imbalanced, let's see what we can do about this:

# **Q:** What does it mean when the data is imbalanced? Why does it need to be balanced?
# - **A:** Imbalanced data means that the distribution of the dataset is not properly balanced in the different classes. In our example, the minority class (0-> healthy, no pd) is underrepresented, accounting for only 25.4% of cases. If we try training the model without first balancing the dataset, the model will be biased against the majority class (1-> patients with PD). Thus, it is crucial to balance the data for higher prediction accuracy
# - **Further reading:** https://medium.com/game-of-bits/how-to-deal-with-imbalanced-data-in-classification-bd03cfc66066

# In[66]:


majority= data[data['class'] == 1]
minority= data[data['class'] == 0]
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=38)
balanced_data = pd.concat([majority, minority_upsampled])
print("Class distribution after balancing the data:")
print(balanced_data['class'].value_counts())


# ### Separating features, splitting the dataset, & feature scaling:

# In[4]:


y = balanced_data.loc[:,'class']
X = balanced_data.drop(['class', 'id'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)


# In[5]:


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# ### Testing different machine learning models:

# **Q: Why do we need to test multiple machine learning models?**
# - **A:** Every model has its own strengths and limitations, when you test different models, you are able to compare between their performances and determine which one is best for your project.

# #### Model 1: SVM RBF

# **Q:** What is a support vector machine (SVM)? 
# - **A:** SVM is a machine learning algorithm that can be used to solve both regression and classification problems. It creates a hyperplane that separates data into classes.
# - **Q:** What is radial basis function (RBF) SVM?
# - **A:** The RBF kernel is a SVM kernel used in classification problems where the dataset cannot be linearly separated.
# - **Further reading:** https://medium.com/analytics-vidhya/introduction-to-svm-and-kernel-trick-part-1-theory-d990e2872ace

# In[51]:


svmmodel = SVC(kernel='rbf')
svmmodel.fit(X_train, y_train)


# In[52]:


pred_test = svmmodel.predict(X_test)
testaccuracy= accuracy_score(y_test, pred_test)
print("SVM RBF model accuracy:", testaccuracy)


# In[53]:


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred_test))


# In[54]:


print(classification_report(y_test, pred_test))


# In[67]:


feature_names=data.columns
feature_importance = pd.DataFrame(feature_names, columns=["feature"])
feature_importance = feature_importance.sort_values(by=['feature'], ascending=False)
feature_importance


# #### Model 2: Logistic regression

# **Q: What is logistic regression?**
# - **A:** Logistic regression is a machine learning algorithm used to solve classification problems. It uses sigmoid (logistic) function to find the relationships between variables.
# - **Further reading:** https://medium.com/axum-labs/logistic-regression-vs-support-vector-machines-svm-c335610a3d16

# In[15]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


# In[16]:


y_pred = logmodel.predict(X_test)
logaccuracy = accuracy_score(y_test, y_pred)
print("Logistic regression accuracy:", logaccuracy)


# In[17]:


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[18]:


print(classification_report(y_test, y_pred))


# In[48]:


feature_names=data.columns
feature_importance = pd.DataFrame(feature_names, columns=["feature"])
feature_importance = feature_importance.sort_values(by=['feature'], ascending=False)
feature_importance


# **Q: What is a confusion matrix? And how do I interpret one?**
# - **A:** A confusion matrix is a table that helps us understand how well a model's predictions perform by showing how many predictions are correct or incorrect per class. In a confusion matrix, columns represent predictions made my the model and rows represent the actual classes (general format).  
# - TP: true positive, indicating the number of positive cases that are correctly
# classified by the classifier.
# - TN: true negative, indicating the number of negative cases that are correctly classified by the classifier.
# - FP: false positive, indicating the number of positive cases that are incorrectly classified by the classifier.
# - FN: false negative, indicating the number of negative cases that are incorrectly classified by the classifier.
# - A good model will have high TP and TN rates and low FP and FN rates.
# - **Further reading:** https://medium.com/analytics-vidhya/what-is-a-confusion-matrix-d1c0f8feda5
# <center> <img src="644aea65cefe35380f198a5a_class_guide_cm08.png" width="500" height="300"> <center>
# 

# ### Result interpretation:

# #### Both models performed well, although logistic regression performed slightly better than SVM RBF, with 91.1% accuracy and 90.2% accuracy, respectively. Both models achieved a high F1 score, again, logistic regression has a slightly higher score than SVM RBF. 

# ### Further reading on Parkinson's disease:
# - Bloem, B. R., Okun, M. S., & Klein, C. (2021). Parkinson’s disease. The Lancet, 397(10291), 2284–2303. https://doi.org/10.1016/S0140-6736(21)00218-X
# - Marino, B. L. B., de Souza, L. R., Sousa, K. P. A., Ferreira, J. V., Padilha, E. C., da Silva, C. H. T. P., Taft, C. A., & Hage-Melim, L. I. S. (2019). Parkinson’s Disease: A Review from Pathophysiology to Treatment. Mini-Reviews in Medicinal Chemistry, 20(9), 754–767. https://doi.org/10.2174/1389557519666191104110908
# - Mei, J., Desrosiers, C., & Frasnelli, J. (2021). Machine Learning for the Diagnosis of Parkinson’s Disease: A Review of Literature. Frontiers in Aging Neuroscience, 13, 633752. https://doi.org/10.3389/FNAGI.2021.633752/BIBTEX
# 
