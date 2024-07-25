#########################################
# GridSearch 
#########################################

# import libraries
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.metrics import r2_score

my_df = pd.read_csv("data/sample_data_regression.csv")

# Split data into input and output
X = my_df.drop(["output"], axis=1)
y = my_df["output"]

# Instanciate RandomForestRegressor
regressor = RandomForestRegressor(random_state=42)

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    shuffle=True, random_state=42)
# Instanciate and GridSearch
gscv = GridSearchCV(estimator=regressor, 
                    param_grid={"n_estimators":[50, 100, 500, 700],
                                "max_depth": [2, 3, 4, 5, 6, 7, 8, 10, None],
                                "min_samples_leaf":[3,4,5,6,7,8,9,10]},
                    cv = 5,
                    scoring="r2", n_jobs= -1)


gscv.fit(X_train, y_train)

print("best training score: ", gscv.best_score_)
print("Best estimator: ", gscv.best_estimator_)

### Assign regressor
regressor = gscv.best_estimator_
# regressor.estimators_
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

r2_score(y_test, y_pred)