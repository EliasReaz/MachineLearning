########################################################################
## Decision Tree Regression - ABC grocery task
########################################################################


########################################################################
## import libraries
########################################################################
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

########################################################################
## import data
########################################################################

# import data
data_for_modeling = pickle.load(open("data/regression_modeling.p", "rb"))
print(data_for_modeling.columns)
# drop uncessary columns
data_for_modeling.drop(["customer_id"], axis=1, inplace=True)

# shuffle data
data_for_modeling = shuffle(data_for_modeling, random_state=42)

########################################################################
## Deal with missing values
########################################################################
print(data_for_modeling.isna().sum())

## only few missing values, so we drop rows with any missing value
data_for_modeling.dropna(how="any", inplace=True)

########################################################################
## Decision Tress can handle outliers
########################################################################

########################################################################
## Split input variables and output variables
########################################################################

X = data_for_modeling.drop("customer_loyalty_score", axis=1)
y = data_for_modeling["customer_loyalty_score"]

########################################################################
## Split out train and test set
########################################################################

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

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
## A high number of features does not affect accuracy of Decison Tree.
## But too many features affect computational time. 
########################################################################


########################################################################
## Model training
########################################################################
regressor = DecisionTreeRegressor(random_state=42, max_depth=4)
regressor.fit(X_train, y_train)
########################################################################
## Model Assessment
########################################################################
## predict on X_test
y_pred = regressor.predict(X_test)

## evaluate r2_score
r_squared = r2_score(y_test, y_pred)
print(f"r_squared: {round(r_squared,4)}")

# Cross validation
cv = KFold(n_splits=4, shuffle=True, random_state=42)
cvscores = cross_val_score(regressor, X_train, y_train, cv=cv, scoring="r2")
print(f"cv mean: {cvscores.mean()}")


# adjusted r2
num_of_data_points, num_of_input_var = X_test.shape
adjusted_r2 = 1 - (1 - r_squared)*(num_of_data_points - 1)/(num_of_data_points - num_of_input_var - 1)
print(f"Adjusted r_squared: {adjusted_r2}")

### looking for optimal max_depth hyperparameter
###################################################
# max_depth_list = list(range(1,9))
# accuracy_list = []

# for depth in max_depth_list:
    
#     regressor = DecisionTreeRegressor(max_depth=depth, random_state=42)
#     regressor.fit(X_train, y_train)
#     y_pred = regressor.predict(X_test)
#     accuracy = r2_score(y_test, y_pred)
#     accuracy_list.append(accuracy)  
    
# max_accuracy = max(accuracy_list)
# max_accuracy_idx = accuracy_list.index(max_accuracy)
# max_depth = max_depth_list[max_accuracy_idx]

# ## plot 
# plt.plot(max_depth_list, accuracy_list)
# plt.scatter(max_depth, max_accuracy, marker="x", color="red")
# plt.xlabel("max depth")
# plt.ylabel("accuracy")
# plt.tight_layout()
# plt.show() 
### looking for optimal max_depth hyperparameter
#################################### 

feature_importance_df = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance_df], axis=1)
feature_importance_summary.columns = ["feature_name", "feature_importance"]
feature_importance_summary.sort_values(by="feature_importance", inplace=True)
print(feature_importance_summary)

# plot barh
plt.barh(feature_importance_summary["feature_name"], feature_importance_summary["feature_importance"])
plt.xlabel("feature importance")
plt.title("Feature importance")
plt.tight_layout()
plt.show()


## plot tree
plt.figure(figsize=(25,15))
tree = plot_tree(regressor, 
                 feature_names=X.columns, 
                 filled=True, 
                 fontsize=16)

