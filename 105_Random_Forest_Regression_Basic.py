
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

my_df = pd.read_csv("data/sample_data_regression.csv")

# Split data into input and output objects
X = my_df.drop(["output"], axis=1)
y = my_df["output"]

# Split data into train and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Instantiate model object
regressor = RandomForestRegressor(random_state=42, n_estimators=1000) 

# train the model
regressor.fit(X_train, y_train)

# Assess model accuracy
y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)

# feature importance
regressor.feature_importances_

feature_importance_df = pd.DataFrame(regressor.feature_importances_)
feature_names_df = pd.DataFrame(X.columns)
summary_feature_importance = pd.concat([feature_names_df, feature_importance_df], axis=1)
summary_feature_importance.columns = ["feature_names", "feature_importance"]
summary_feature_importance.sort_values(by="feature_importance", inplace=True)
summary_feature_importance.reset_index(drop=True, inplace=True)
print(summary_feature_importance)

plt.barh(summary_feature_importance["feature_names"], 
         summary_feature_importance["feature_importance"])
plt.xlabel("Feature importance")
plt.tight_layout()
plt.show()