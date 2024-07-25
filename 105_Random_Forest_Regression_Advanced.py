########################################################################
## Random Forest Regression - ABC grocery task
########################################################################
########################################################################
## import libraries
########################################################################
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

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
## Regreesion Tress can handle outliers
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
regressor = RandomForestRegressor(random_state=42)
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
####################################################
num_of_data_points, num_of_input_var = X_test.shape
adjusted_r2 = 1 - (1 - r_squared)*(num_of_data_points - 1)/(num_of_data_points - num_of_input_var - 1)
print(f"Adjusted r_squared: {adjusted_r2}")

#####################################################
## Feature Importance
#####################################################
feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names, feature_importance], axis=1)
feature_importance_summary.columns = ["feature_name", "feature_importance"]
feature_importance_summary.sort_values(by="feature_importance", inplace=True)
print(feature_importance_summary)

## plot barh
plt.barh(feature_importance_summary["feature_name"], 
         feature_importance_summary["feature_importance"])
plt.xlabel("Feature importance")
plt.title("Feature Importance Summary")
plt.tight_layout()
plt.show()

#######################################################
### Now see permutation importance
#######################################################
permutation_imp = permutation_importance(regressor, X_test, y_test, random_state=42)
feature_permutation_importance = pd.DataFrame(permutation_imp["importances_mean"])
feature_names = pd.DataFrame(X.columns)
feature_permutation_summary = pd.concat([feature_names, feature_permutation_importance], axis=1)
feature_permutation_summary.columns = ["feature_name", "feature_importance"]
feature_permutation_summary.sort_values(by="feature_importance", inplace=True)
print(feature_permutation_summary)

## plot barh
plt.barh(feature_permutation_summary["feature_name"], 
         feature_permutation_summary["feature_importance"])
plt.xlabel("Feature (permutation) importance mean")
plt.title("Feature Permutation Importance Summary")
plt.tight_layout()
plt.show()

## begin - prediction under hood
y_pred[0]
new_data = [X_test.iloc[0]]
regressor.estimators_

predictions = []
tree_count = 0
for tree in regressor.estimators_:
    prediction = tree.predict(new_data)[0]
    predictions.append(prediction)
    tree_count += 1
print(predictions)
sum(predictions)/tree_count
## end - prediction under hood

## save the model in data folder
pickle.dump(regressor, open("data/random_forest_regressor.p", "wb"))
pickle.dump(one_hot_encoder, open("data/random_forest_regressor_ohe.p", "wb"))
