########################################################################
## Decision Tree Classification - ABC grocery task
########################################################################


########################################################################
## import libraries
########################################################################
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
# Recursive Feature Elemination
from sklearn.preprocessing import OneHotEncoder

########################################################################
## import data
########################################################################

# import data
data_for_modeling = pd.read_pickle("data/abc_classification_modelling.p")
print(data_for_modeling.columns)
print(data_for_modeling.head())
# drop uncessary columns
data_for_modeling.drop(["customer_id"], axis=1, inplace=True)

# shuffle data
data_for_modeling = shuffle(data_for_modeling, random_state=42)

#############################
## see class balance
#############################
data_for_modeling["signup_flag"].value_counts(normalize=True)

########################################################################
## Deal with missing values
########################################################################
print(data_for_modeling.isna().sum())

## only few missing values, so we drop rows with any missing value
data_for_modeling.dropna(how="any", inplace=True)

########################################################################
## Split input variables and output variables
########################################################################

X = data_for_modeling.drop("signup_flag", axis=1)
y = data_for_modeling["signup_flag"]

########################################################################
## Split out train and test set
########################################################################

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, 
                                                    random_state=42, stratify=y)

########################################################################
## Deal with categorical features
########################################################################
one_hot_encoder  = OneHotEncoder(sparse_output=False, drop='first')

categorical_vars = ["gender"]
X_train_encoder_array = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoder_array = one_hot_encoder.transform(X_test[categorical_vars])

# print("encoder variables array")
# print(encoder_vars_array)

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)
print("encoder feature names")
print(encoder_feature_names)

X_train_encoder = pd.DataFrame(X_train_encoder_array, columns=encoder_feature_names)
## concat df side by side column wise
X_train = pd.concat([X_train.reset_index(drop=True),
                   X_train_encoder.reset_index(drop=True)], axis=1)
X_train.drop(categorical_vars, axis=1, inplace=True)

X_test_encoder = pd.DataFrame(X_test_encoder_array, columns=encoder_feature_names)
## concat df side by side column wise
X_test = pd.concat([X_test.reset_index(drop=True),
                   X_test_encoder.reset_index(drop=True)], axis=1)
X_test.drop(categorical_vars, axis=1, inplace=True)

########################################################################
## Model training
########################################################################
clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)

########################################################################
## Model Assessment
########################################################################
## predict on X_test
y_pred_class = clf.predict(X_test)

y_pred_prob = clf.predict_proba(X_test)[:,1]

# Confusin matrix

conf_matrix = confusion_matrix(y_test, y_pred_class)
print(conf_matrix)

# plt.style.use("seaborn")
plt.matshow(conf_matrix, cmap="coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
for (i,j), corr_val in np.ndenumerate(conf_matrix):
    plt.text(j,i, corr_val, ha="center", va="center", fontsize=20)
plt.show()

# Accuracy: the number of correct classification out of all attempted classifications
accuracy_score(y_test, y_pred_class)

# Precision: Out of all predicted as positive, how many are actually positive 
precision_score(y_test, y_pred_class)

# Recall: Out of all actual positive classes, how many are predicted as positive  
recall_score(y_test, y_pred_class)

# F1 score: The harmonic mean of precision and recall
f1_score(y_test, y_pred_class)

#######################################################################
# finding optimal max_depth
########################################################################

max_depth_list = list(range(1,15))
accuracy_list = []

for depth in max_depth_list:
    
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test, y_pred)
    accuracy_list.append(accuracy)  
    
max_accuracy = max(accuracy_list)
max_accuracy_idx = accuracy_list.index(max_accuracy)
max_depth = max_depth_list[max_accuracy_idx]

## plot 
plt.plot(max_depth_list, accuracy_list)
plt.scatter(max_depth, max_accuracy, marker="x", color="red")
plt.xlabel("max depth")
plt.ylabel("accuracy (F1)")
plt.tight_layout()
plt.show() 


## plot decision tree
plt.figure(figsize=(25,15))
tree = plot_tree(clf, 
                 feature_names=X.columns, 
                 filled=True, 
                 fontsize=16)
