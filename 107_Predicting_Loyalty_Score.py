# import required libraries
import pandas as pd
import pickle

# import model
#########################################

random_forest_regressor = pickle.load(open("data/random_forest_regressor.p", "rb"))
one_hot_encoder = pickle.load(open("data/random_forest_regressor_ohe.p", "rb"))

# import customers for scoring
#########################################
to_be_scored = pickle.load(open("data/regression_scoring.p", "rb"))

# drop unused columns
#########################################
customer_id_list = to_be_scored["customer_id"]
to_be_scored.drop(["customer_id"], axis=1, inplace=True)

# drop or impute missing values
#########################################
to_be_scored.dropna(how="any", inplace=True)

#########################################
# Apply one hot encoding
#########################################
categorical_vars = ["gender"]
encoder_vars_array = one_hot_encoder.transform(to_be_scored[categorical_vars])
# print("encoder variables array")
# print(encoder_vars_array)

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)
print("encoder feature names")
print(encoder_feature_names)

encoder_var_df = pd.DataFrame(encoder_vars_array, columns=encoder_feature_names)

## concat df side by side column wise
to_be_scored = pd.concat([to_be_scored.reset_index(drop=True),
                   encoder_var_df.reset_index(drop=True)], axis=1)
to_be_scored.drop(categorical_vars, axis=1, inplace=True)
print(to_be_scored.head())

##############################################
# Make prediction
##############################################

loyalty_predictions = random_forest_regressor.predict(to_be_scored)

## present scores along with customer id
# loyalty_predictions_df = pd.DataFrame(loyalty_predictions)
# customer_ids = pd.DataFrame(customer_id_list)
# loyalty_prediction_by_customer_id = pd.concat([customer_ids, loyalty_predictions_df], axis=1)
# loyalty_prediction_by_customer_id.columns = ["customer_id", "loyalty_predicted"]
# print("")
# print("-------------------")
# print("predicted loyalty")
# print("-------------------")
# print(loyalty_prediction_by_customer_id.head())