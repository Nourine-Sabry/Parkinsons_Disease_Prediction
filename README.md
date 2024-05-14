# **Parkinson's disease prediction using machine learning**
### **Project overview:**
Parkinson's disease (PD) is the second most common progressive neurodegenerative disease. PD symptoms include tremor, rigidity, cognitive impairment, and gastrointestinal issues. Diagnosing PD often relies on medical observations of motor symptoms, but in cases of early non-motor symptoms, subtle and mild symptoms might be overlooked. The aim of this project is to implement a machine learning model to try and predict whether a patient has PD or not based on speech features.
### **Dataset overview:**
The dataset used in this project was obtained from the following Kaggle link: https://www.kaggle.com/datasets/dipayanbiswas/parkinsons-disease-speech-signal-features/data
This dataset consists of various speech signal features extracted from patients with PD and healthy individuals. The target variable ‘class’ indicates the presence (1) or absence (0) of PD. It consists of 756 rows and 755 columns. While it does not have duplicate values or missing data, it has highly correlated features and the data is also imbalanced.
### **Quick video demonstration:**
https://nileuniversity-my.sharepoint.com/:v:/g/personal/n_mamdouh2198_nu_edu_eg/ETLCeNe49gpNmefMsw5OxpkBnDsNFo_NHd3TJTxUsr0WGQ?e=sxmO0y
### **Streamlit link**
http://192.168.51.35:8501/
### **Steps:**
* Importing libraries
* Importing the dataset
* Exploring the dataset & its features
* EDA & data visualization
  * This step showed that the data is imbalanced and contains highly correlated features
* Removing highly correlated features
* Balancing the data
* Separating features, splitting the dataset, & feature scaling
* Training and comparing between two different models (SVM RBF & Logistic regression)
  * SVM RBF accuracy: 90.2%
  * Logistic regression: 91.1%
* Model deployment using streamlit
### Results:
While both SVM RBF and logistic regression performed good, logistic regression performed slightly better than SVM RBF, with 91.1% accuracy and 90.2% accuracy, respectively. Both models achieved a high F1 score, again, logistic regression has a slightly higher score than SVM RBF. Overall, logistic regression outperformed SVM RBF in all evaluation metrics used in this project.
### Limitations:
* Two models were trained in this project, additional models could have been trained.
### Conclusion:
This is one of many in a large sea of machine learning projects that have shown the power of machine learning in solving real-world problems such as diagnostics, while no model is 100% accurate, machine learning has greatly impacted numerous fields such as healthcare, finance, and advertising. It has also shown that it is important to test multiple machine learning models, because it enables comparing between different models and seeing which one performs better than the others, which will help in deploying more accurate and efficient models. In this project, logistic regression slightly outperformed SVM RBF. This is probably due to the fact that logistic regression performs better on linearly separable data. 
### Further reading on Parkinson's disease:
* Bloem, B. R., Okun, M. S., & Klein, C. (2021). Parkinson’s disease. The Lancet, 397(10291), 2284–2303. https://doi.org/10.1016/S0140-6736(21)00218-X
* Marino, B. L. B., de Souza, L. R., Sousa, K. P. A., Ferreira, J. V., Padilha, E. C., da Silva, C. H. T. P., Taft, C. A., & Hage-Melim, L. I. S. (2019). Parkinson’s Disease: A Review from Pathophysiology to Treatment. Mini-Reviews in Medicinal Chemistry, 20(9), 754–767. https://doi.org/10.2174/1389557519666191104110908
* Mei, J., Desrosiers, C., & Frasnelli, J. (2021). Machine Learning for the Diagnosis of Parkinson’s Disease: A Review of Literature. Frontiers in Aging Neuroscience, 13, 633752. https://doi.org/10.3389/FNAGI.2021.633752/BIBTEX
### Built with:
* Matplotlib
* Seaborn
* Numpy
* Pandas
* Sklearn
### Authors:
Nourine Sabry
