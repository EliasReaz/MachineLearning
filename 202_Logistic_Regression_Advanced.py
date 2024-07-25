########################################################################
## Logistic Regression - ABC grocery task
########################################################################


########################################################################
## import libraries
########################################################################
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
# Recursive Feature Elemination
from sklearn.feature_selection import RFECV 
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

## Check class balance
data_for_modeling["signup_flag"].value_counts(normalize=True)

########################################################################
## Deal with missing values
########################################################################
print(data_for_modeling.isna().sum())

## only few missing values, so we drop rows with any missing value
data_for_modeling.dropna(how="any", inplace=True)

########################################################################
## Deal with outliers
########################################################################
outliers_investigation = data_for_modeling.describe()
print(outliers_investigation)
# it seems distance_from_store, total_sales, total_items, and average_basket_value
# have outliers
# draw boxplot excluding categorical variable "gender" to see outliers
data_for_modeling.drop("gender", axis=1).plot(kind="box", vert=False)

outlier_columns = ["distance_from_store", "total_sales", "total_items"]
## boxplot approach
#######################
for column in outlier_columns:
    
    lower_quartile = data_for_modeling[column].quantile(0.25)
    higher_quantile = data_for_modeling[column].quantile(0.75)
    iqr = higher_quantile - lower_quartile
    iqr_extended = iqr * 2
    min_border = lower_quartile - iqr_extended
    max_border = higher_quantile + iqr_extended
    
    
    outliers = data_for_modeling[(data_for_modeling[column]< min_border) | (data_for_modeling[column]> max_border)].index
    print(f"{len(outliers)} outliers are detected in column {column}")
    
    data_for_modeling.drop(outliers, inplace=True)

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
## Feature Selection
########################################################################
clf = LogisticRegression(random_state=42, max_iter=1000)
feature_selector = RFECV(clf)

fit = feature_selector.fit(X_train, y_train)
print("cross-validation results cv_results_", fit.cv_results_["mean_test_score"])
fit.get_feature_names_out()
optimal_feature_count = fit.n_features_
print(f"Optimal number of features: {optimal_feature_count}")

X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]

## plot score w.r.t number of features
plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()
plt.show()

########################################################################
## Model training
########################################################################
clf = LogisticRegression(random_state=42, max_iter=1000)
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

# Accuracy: The number of correct classification out of all attempted classifications
accuracy_score(y_test, y_pred_class)

# Precision: Out of all predicted as positive, how many are actually positive 
precision_score(y_test, y_pred_class)

# Recall: Out of all actual positive classes, how many are predicted as positive  
recall_score(y_test, y_pred_class)

# F1 score: The harmonic mean of precision and recall
f1_score(y_test, y_pred_class)

#######################################################################
# finding optimal threshold
########################################################################

thresholds = np.arange(0, 1, 0.01)
precisionscores = []
recallscores = []
f1scores = []

for threshold in thresholds:
    pred_class = (y_pred_prob > threshold)*1
    precision = precision_score(y_test, pred_class, zero_division=0) # zero_division=0 no warning
    precisionscores.append(precision)
    
    recall = recall_score(y_test, pred_class)
    recallscores.append(recall)
    
    f1score = f1_score(y_test, pred_class)
    f1scores.append(f1score)
    
max_f1 = max(f1scores)
max_f1_idx = f1scores.index(max_f1)
# max_f1_idx
threshold_at_max_f1 = thresholds[max_f1_idx]
# threshold_at_max_f1
# max_f1

plt.plot(thresholds, precisionscores, label="Precision", linestyle="--")
plt.plot(thresholds, recallscores, label="Recall", linestyle="--")
plt.plot(thresholds, f1scores, label="F1", linewidth=5)
plt.xlabel("Classification threshold probability")
plt.ylabel("Performance Score")
plt.title(f"Finding an Optimal Threshold for Classification Model \n Max F1 {round(max_f1,2)}: (Threshold = {round(threshold_at_max_f1,2)})")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

threshold_optimal = threshold_at_max_f1
y_pred_class_opt_threshold = (y_pred_prob > threshold_optimal) *1
