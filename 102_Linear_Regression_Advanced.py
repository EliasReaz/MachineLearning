########################################################################
## Linear Regression - ABC grocery task
########################################################################


########################################################################
## import libraries
########################################################################
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.utils import shuffle
from sklearn.feature_selection import RFECV # Recursive Feature Elemination
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
## Deal with outliers
########################################################################
outliers_investigation = data_for_modeling.describe()
print(outliers_investigation)
# it seems distance_from_store, total_sales, total_items, and average_basket_value
# have outliers
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
## Feature Selection
########################################################################
regressor = LinearRegression()
feature_selector = RFECV(regressor, scoring="r2")

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
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.coef_)
########################################################################
## Model Assessment
########################################################################
## predict on X_test
y_pred = regressor.predict(X_test)

## evaluate r2_score
r_squared = r2_score(y_test, y_pred)

# Cross validation
cv = KFold(n_splits=4, shuffle=True, random_state=42)
cvscores = cross_val_score(regressor, X_train, y_train, cv=cv, scoring="r2")
print(f"cv mean: {cvscores.mean()}")
print(regressor.coef_)

# adjusted r2
num_of_data_points, num_of_input_var = X_test.shape
adjusted_r2 = 1 - (1 - r_squared)*(num_of_data_points - 1)/(num_of_data_points - num_of_input_var - 1)
print(f"Adjusted r_squared: {adjusted_r2}")

# extract coefficient
coefficients = pd.DataFrame(regressor.coef_)
input_variable_names = pd.DataFrame(X_train.columns)
summary_stats = pd.concat([input_variable_names, coefficients], axis=1)
summary_stats.columns = ["input_variable", "coefficient"]
print(summary_stats)

# Extract model intercept
intercept = regressor.intercept_
print(intercept)
