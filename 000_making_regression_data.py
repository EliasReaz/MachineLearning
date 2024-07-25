import pandas as pd
import pickle

loyalty_score = pd.read_excel("data/grocery_database.xlsx", sheet_name="loyalty_scores")
customer_details = pd.read_excel("data/grocery_database.xlsx", sheet_name="customer_details")
transactions = pd.read_excel("data/grocery_database.xlsx", sheet_name="transactions")

print(transactions["customer_id"].nunique())
print(loyalty_score["customer_id"].nunique())
print(customer_details["customer_id"].nunique())

print(transactions["transaction_date"].min(), transactions["transaction_date"].max())

print(transactions.head(3))


grocery_data = pd.merge(left=customer_details, right=loyalty_score, 
                        how="left", on="customer_id")
print(grocery_data.shape)

print(grocery_data.head())

print(grocery_data.isna().sum())

sales_summary = transactions.groupby("customer_id").agg({"sales_cost":"sum", 
                                        "num_items":"sum", 
                                        "transaction_id":"count", 
                                        "product_area_id":"nunique"}).reset_index()
print(sales_summary.head())

sales_summary.columns = ["customer_id", "total_sales", "total_items", 
                         "transaction_count", "product_area_count"]

## Assumption is that customers with higher mean value per transaction is more loyal 
sales_summary["average_basket_value"] = sales_summary["total_sales"]/sales_summary["transaction_count"]
print(sales_summary.head())

## merge sales_summary with grocery_data
data_for_regression= pd.merge(grocery_data, sales_summary, how="inner", on="customer_id")
print(data_for_regression.tail())

regression_modeling = data_for_regression.loc[data_for_regression["customer_loyalty_score"].notna()]
regression_scoring = data_for_regression.loc[data_for_regression["customer_loyalty_score"].isna()]

regression_scoring.drop(["customer_loyalty_score"], axis=1, inplace=True)

pickle.dump(regression_modeling, open("data/regression_modeling.p", "wb"))
pickle.dump(regression_scoring, open("data/regression_scoring.p", "wb"))


""" 
I tried in a different way
###################################################################################
tran_pivot = pd.pivot_table(transactions, index="customer_id", columns="product_area_id",
                            values="sales_cost", aggfunc="sum", fill_value=0,
                            margins=True, margins_name="total_sales").rename_axis(None, axis=1)
print(tran_pivot.head())


sales_percent_per_product = tran_pivot.div(tran_pivot["total_sales"], axis=0)
sales_percent_per_product.drop(columns="total_sales", axis=1, inplace=True)
sales_percent_per_product.columns = ["nonfood_sales_percent", "vegetables_sales_percent", "fruits_sales_percent",
                                     "dairy_sales_percent","meat_sales_percent"]
print(sales_percent_per_product.head())
sales_percent_per_product.reset_index(drop=False, inplace=True)
print(sales_percent_per_product.customer_id.nunique())

print(sales_percent_per_product.head())

grocery_data_for_regression = pd.merge(grocery_data, sales_percent_per_product, how="inner", on="customer_id")
print(grocery_data_for_regression.shape)
print(grocery_data_for_regression.head())

print(grocery_data_for_regression.isna().sum())
## look at nan values
grocery_data_for_regression[grocery_data_for_regression["gender"].isna()].head(10)

## drop rows with all nan in distance_from_store, credit_score, customer_loyalty_score
grocery_data_for_regression.dropna(axis="index", how="all", 
                                   subset=["distance_from_store", "credit_score", "customer_loyalty_score"],
                                   inplace=True)

print(grocery_data_for_regression.isna().sum())

grocery_data_for_regression.drop(["gender"], axis="columns").corr()
"""

