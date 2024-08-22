#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# In[2]:


pima_df = pd.read_csv("cleaned_data.csv")
pima_df


# In[3]:


array = pima_df.values
X = pima_df.iloc[:,0:8]
y = pima_df.iloc[:,8]
#X = array[:,0:8] # select all rows and first 8 columns which are the attributes
#Y = array[:,8]   # select all rows and the 8th column which is the classification "Yes", "No" for diabeties
test_size = 0.30 # taking 70:30 training and test set
seed =1 # Random numbmer seeding for reapeatability of the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


# In[4]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score


# In[5]:


models = {
    "Logisitic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train)  # Train model

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Training set performance
    model_train_accuracy = accuracy_score(y_train, y_train_pred)  # Calculate accuracy
    model_train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    model_train_precision = precision_score(y_train, y_train_pred)
    model_train_recall = recall_score(y_train, y_train_pred)
    model_train_rocauc_score = roc_auc_score(y_train, y_train_pred)

    # Test set performance
    model_test_accuracy = accuracy_score(y_test, y_test_pred)  # Calculate accuracy
    model_test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    model_test_precision = precision_score(y_test, y_test_pred)
    model_test_recall = recall_score(y_test, y_test_pred)
    model_test_rocauc_score = roc_auc_score(y_test, y_test_pred)

    print(list(models.keys())[i])

    print('Model performance for Training set')
    print('- Accuracy: {:.4f}'.format(model_train_accuracy))
    print('- F1 score: {:.4f}'.format(model_train_f1))

    print('- Precision: {:.4f}'.format(model_train_precision))
    print('- Recall: {:.4f}'.format(model_train_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_train_rocauc_score))


    print('----------------------------------')


    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy))
    print('- F1 score: {:.4f}'.format(model_test_f1))

    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score))

    print('='*35)
    print('\n')


# In[6]:


# # Hyper parameter training

# rf_params = {"max_depth": [5, 8, 15, None, 10],
#              "max_feature": [5,7, "auto", 8],
#              "min_samples_split": [2, 8, 15, 20],
#              "n_estimators": [100, 200, 500, 1000]}


# In[7]:


#rf_params


# In[8]:


# # Model list for hyperparameter tuning
# randomcv_models = [
#     ("RF", RandomForestClassifier(), rf_params)
# ]


# In[9]:


#randomcv_models


# In[10]:


# from sklearn.model_selection import RandomizedSearchCV

# model_param = {}
# for name, model, params in randomcv_models:
#     random = RandomizedSearchCV(estimator=model,
#                                 param_distributions=params,
#                                 n_iter=100,
#                                 cv=3,
#                                 verbose=2,
#                                 n_jobs=-1)

#     random.fit(X_train, y_train)
#     model_param[name] = random.best_params_

# for model_name in model_param:
#     print(f"---------------- Best Params for {model_name} ----------------")
#     print(model_param[model_name])


# In[11]:


rf_params = {
    "max_depth": [5, 8, 15, None, 10],
    "max_features": [5, 7, "auto", 8],  # corrected parameter name
    "min_samples_split": [2, 8, 15, 20],
    "n_estimators": [100, 200, 500, 1000]
}

# Define the hyperparameter grid for DecisionTreeClassifier
dt_params = {
    "max_depth": [5, 10, 15, None],
    "max_features": [None, "auto", "sqrt", "log2"],
    "min_samples_split": [2, 10, 20],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}


# In[12]:


rf_params


# In[13]:


dt_params


# In[14]:


# Model list for hyperparameter tuning
randomcv_models = [
    ("RF", RandomForestClassifier(), rf_params),
    ("dt", DecisionTreeClassifier(), dt_params)
]


# In[15]:


randomcv_models


# In[16]:


from sklearn.model_selection import RandomizedSearchCV

model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator=model,
                                param_distributions=params,
                                n_iter=100,
                                cv=3,
                                verbose=2,
                                n_jobs=-1)

    random.fit(X_train, y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f"---------------- Best Params for {model_name} ----------------")
    print(model_param[model_name])


# In[17]:


models = {
    "Decision Tree": DecisionTreeClassifier(min_samples_split=20, max_depth=5, min_samples_leaf=5, max_features='log2', criterion='gini'),
    "Random Forest": RandomForestClassifier(n_estimators = 500, min_samples_split=15, max_features=8, max_depth=5)
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train)  # Train model

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Training set performance
    model_train_accuracy = accuracy_score(y_train, y_train_pred)  # Calculate accuracy
    model_train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    model_train_precision = precision_score(y_train, y_train_pred)
    model_train_recall = recall_score(y_train, y_train_pred)
    model_train_rocauc_score = roc_auc_score(y_train, y_train_pred)

    # Test set performance
    model_train_accuracy = accuracy_score(y_test, y_test_pred)  # Calculate accuracy
    model_train_f1 = f1_score(y_test, y_test_pred, average='weighted')
    model_train_precision = precision_score(y_test, y_test_pred)
    model_train_recall = recall_score(y_test, y_test_pred)
    model_train_rocauc_score = roc_auc_score(y_test, y_test_pred)

    print(list(models.keys())[i])

    print('Model performance for Training set')
    print('- Accuracy: {:.4f}'.format(model_train_accuracy))
    print('- F1 score: {:.4f}'.format(model_train_f1))

    print('- Precision: {:.4f}'.format(model_train_precision))
    print('- Recall: {:.4f}'.format(model_train_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_train_rocauc_score))


    print('----------------------------------')


    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(model_test_accuracy))
    print('- F1 score: {:.4f}'.format(model_test_f1))

    print('- Precision: {:.4f}'.format(model_test_precision))
    print('- Recall: {:.4f}'.format(model_test_recall))
    print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score))

    print('='*35)
    print('\n')


# In[18]:


## Plot ROC AUC Curve
from sklearn.metrics import roc_auc_score, roc_curve
plt.figure()

# Add the models to the list that you want to view on the ROC plot
auc_models = [
{
    'label': 'Random Forest Classifier',
    'model': RandomForestClassifier(n_estimators=500, min_samples_split=15,
                                    max_features=8, max_depth=5),
    'auc': 0.7462
},
]

# Create loop through all models
# create loop through all model
for algo in auc_models:
    model = algo['model'] # select the model
    model.fit(X_train, y_train) # train the model
    
    # Compute False positive rate, and True positive rate
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    
    # Calculate Area under the curve to display on the plot
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (algo['label'], algo['auc']))

# Custom settings for the plot
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("auc.png")
plt.show()


# In[19]:


## Plot ROC AUC Curve
from sklearn.metrics import roc_auc_score, roc_curve
plt.figure()

# Add the models to the list that you want to view on the ROC plot
auc_models = [
{
    'label': 'Decission tree Classifier',
    'model': DecisionTreeClassifier(min_samples_split=20, max_depth=5, 
                                    min_samples_leaf=5, max_features='log2', criterion='gini'),
    'auc': 0.7462
},
]

# Create loop through all models
# create loop through all model
for algo in auc_models:
    model = algo['model'] # select the model
    model.fit(X_train, y_train) # train the model
    
    # Compute False positive rate, and True positive rate
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    
    # Calculate Area under the curve to display on the plot
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (algo['label'], algo['auc']))

# Custom settings for the plot
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("auc.png")
plt.show()


# In[20]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Train the RandomForestClassifier and DecisionTreeClassifier with the best hyperparameters
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, max_features=7, min_samples_split=8)
dt_model = DecisionTreeClassifier(min_samples_split=20, max_depth=5, min_samples_leaf=5, max_features='log2', criterion='gini')

# List to hold models and their labels
auc_models = [
    {'label': 'Random Forest Classifier', 'model': rf_model},
    {'label': 'Decision Tree Classifier', 'model': dt_model}
]

# Plot ROC curves for each model
plt.figure()

for algo in auc_models:
    model = algo['model']  # Select the model
    model.fit(X_train, y_train)  # Train the model
    
    # Predict probabilities and compute ROC curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # Calculate AUC
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Plot the ROC curve
    plt.plot(fpr, tpr, label='%s ROC (AUC = %0.2f)' % (algo['label'], auc_score))

# Custom settings for the plot
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("auc.png")
plt.show()


# In[ ]:




