# -*- coding: utf-8 -*-
"""
Created on Thu Jul  29 21:28:06 2021

@author: Vishal pc
"""
import pandas as pd

# Import data from .csv file
filename = 'student-por.csv'
data_Vishal=pd.read_csv(filename)
data_Vishal.shape
data_Vishal.info()
data_Vishal.head()

# 2a.Check the names and types of columns
data_Vishal.columns
data_Vishal.dtypes

# b.Check the missing values
data_Vishal.isnull().sum().sum()
print("Missing values column wise",data_Vishal.isnull().sum())  # column wise

# c.Check the statistics of the numeric fields (mean, min, max, median, count,..etc.)
data_Vishal.describe()
data_Vishal.median()

# d.Check the categorical values
s = (data_Vishal.dtypes == 'object')
object_cols = list(s[s].index)
object_cols_values = list(s[s].value_counts())
print("List of Categorical variables:",object_cols)
print("Count of Categorical variables:",object_cols_values)

#3. Create a new target variable i.e. column name it pass_fristname, which will store the following per row:
data_Vishal.loc[data_Vishal['G1']+data_Vishal['G2']+data_Vishal['G3'] >= 35, 'pass_Vishal'] = '1'  
data_Vishal.loc[data_Vishal['G1']+data_Vishal['G2']+data_Vishal['G3'] <  35, 'pass_Vishal'] = '0'  
data_Vishal.pass_Vishal.head()

#4. Drop the columns G1, G2, G3 permanently.
data_Vishal.drop(columns=['G1','G2','G3'], inplace=True)
data_Vishal.columns

#5. Separate the features from the target variable (class)
features_Vishal = data_Vishal.drop('pass_Vishal', axis=1)
target_Vishal=data_Vishal['pass_Vishal']
features_Vishal.columns

#6. Print out the total number of instances in each class and note into your report and explain your findings in terms of balanced and un-balanced.
print("\nDistribution of values in class:",'\n',data_Vishal['pass_Vishal'].value_counts())

#7.Create two lists one to save the names of your numeric fields and on to save the names of your categorical fields
numeric_features_Vishal=data_Vishal.select_dtypes(include=['int64'])
cat_features_Vishal=["school","sex","address","famsize","Pstatus","Mjob","Fjob","reason","guardian","schoolsup","famsup","paid","activities","nursery","higher","internet","romantic"]

#8.Prepare a column transformer to handle all the categorical variables and convert them into numeric
#values using one-hot encoding.
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
transformer_Vishal=ColumnTransformer(transformers=[("category",OneHotEncoder(), cat_features_Vishal)])

#9. Prepare a classifier decision tree model 
from sklearn.tree import DecisionTreeClassifier
clf_Vishal=DecisionTreeClassifier(criterion="entropy", max_depth = 5)
 
#10.The pipeline should have two steps the first the column transformer you prepared in step 8 and the second the model you prepared in step 9
from sklearn.pipeline import Pipeline
pipeline_Vishal=Pipeline([('Transformer1',transformer_Vishal),('Transformer2',clf_Vishal)])

#11.Split your data into train 80% train and 20% test
from sklearn.model_selection import train_test_split
X_train_Vishal,X_test_Vishal,y_train_Vishal,y_test_Vishal=train_test_split(features_Vishal,target_Vishal,test_size = 0.2, random_state = 77)

#Build classification models
#12. Fit the training data to the pipeline you built in step #11.
pipeline_Vishal.fit(X_train_Vishal,y_train_Vishal)

#13. Cross validate the output on the training data using 10-fold cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#cv = KFold(n_splits=10, shuffle=True, random_state=77)
cv10=cross_val_score(pipeline_Vishal,X_train_Vishal,y_train_Vishal, cv=KFold(n_splits=10, shuffle=True, random_state=77))
print(cv10)

#14. Print out the ten scores and the mean of the ten scores
print("Mean of 10 scores:",cv10.mean())

#15. Visualize the tree using Graphviz
import matplotlib.pyplot as plt
from sklearn import tree
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(clf_Vishal,filled = True);
fig.savefig('decisiontree.png')


#Plot using graphviz
import graphviz 

dot_data = tree.export_graphviz(clf_Vishal, out_file=None, 
                      feature_names=features_Vishal,  
                      class_names=target_Vishal,  
                      filled=True, rounded=True,  
                      special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render("studentPerformance")
graph 

#18. Print out two accuracy score one for the model on the training set and testing set
from sklearn import metrics
y_train_pred = pipeline_Vishal.predict(X_train_Vishal)
accuracyScore = metrics.accuracy_score(y_train_Vishal,y_train_pred)
print("accuracy of model on training set",accuracyScore)

#testing set accuracy score
y_test_pred = pipeline_Vishal.predict(X_test_Vishal)
accuracyScore = metrics.accuracy_score(y_test_Vishal,y_test_pred)
print("accuracy of model on testing set",accuracyScore)
#19. Use the model to predict the test data and printout the accuracy, precision and recall scores and the confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print("\nConfusion matrix:\n",confusion_matrix(y_test_Vishal, y_test_pred))
print("\nClassification Report:\n",classification_report(y_test_Vishal, y_test_pred))

#Fine tune the model 
#20.Using Randomized grid search fine tune your model 
# set of parameters to test
from sklearn.model_selection import RandomizedSearchCV
parameters = {
              "Transformer2__min_samples_split": range(10,300,20),
              "Transformer2__max_depth": range(1,30,2),
              "Transformer2__min_samples_leaf": range(1,15,3)
              #"max_leaf_nodes": [None, 5, 10, 20],
              }

randomized_search_Vishal = RandomizedSearchCV(estimator= pipeline_Vishal,scoring='accuracy',param_distributions=parameters,cv=5,n_iter = 7,refit=True ,verbose=3)

#21. Fit your training data to the gird search object
randomized_search_Vishal.fit(X_train_Vishal,y_train_Vishal)
#22. Print out the best parameters and note them it in your written response
best_params=randomized_search_Vishal.best_params_
print('Best Parameters are:',best_params)
randomized_search_Vishal.best_params_
#23. Print out the score of the model and note it in your written response compare this score with original score you generated in step #14 is it better or worse and explain why.
print("Accuracy:",metrics.accuracy_score(y_test_Vishal, y_test_pred))

#24. Printout the best estimator and note it in your written response
best_estimator = randomized_search_Vishal.best_estimator_
print("Best estimators:",best_estimator)
#25. Fit the test data using the fine-tuned model identified during grid search 
best_model = {'estimator:' : best_estimator}
randomized_search_Vishal.fit(X_test_Vishal,y_test_Vishal)
#26. Printout the precision, re_call and accuracy.
grid_predict = randomized_search_Vishal.predict(X_test_Vishal)
classificationReport = classification_report( y_test_Vishal,randomized_search_Vishal.predict(X_test_Vishal))
print("Classification Report:",classificationReport)
#27. Save the model using the joblib (dump).
import joblib
joblib.dump(best_model,"best_model_Vishal.pkl")
#28. Save the full pipeline using the joblib – (dump).
joblib.dump(pipeline_Vishal,open("pipeline_Vishal.pkl", 'wb'))
