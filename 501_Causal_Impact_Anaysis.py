##########################################
# Causal Impact Anaysis
##########################################
# import required packages

from causalimpact import CausalImpact
import pandas as pd
import numpy as np

## import data 
transactions = pd.read_excel("data/grocery_database.xlsx", sheet_name="transactions")
campaign_data = pd.read_excel("data/grocery_database.xlsx", sheet_name="campaign_data")

## on transactions table: groupby customer_id, transaction_date
sales_per_day = transactions.groupby(["customer_id", "transaction_date"])["sales_cost"].sum().reset_index()
sales_per_day.head()
## merge sales_summary and campaign_data
signup_sales= pd.merge(sales_per_day, campaign_data, how="inner", on="customer_id")
print(signup_sales.shape)
print(signup_sales.head())

## groupby customer_id and transaction_date
signup_sales_mean = signup_sales.pivot_table(index="transaction_date", 
                                                 columns="signup_flag",
                                                 values="sales_cost",
                                                 aggfunc="mean")
print(signup_sales_mean.index)
# make the frequency daily
signup_sales_mean.index.freq = "D"

## swap the impacted column first
signup_sales_mean = signup_sales_mean[[1,0]]
# print(signup_sales_mean.head())
# rename columns
signup_sales_mean.columns=["member", "non_member"]
print(signup_sales_mean.head())

## Apply CausalImpact
pre_period = ["2020-04-01", "2020-06-30"]
post_period = ["2020-07-01", "2020-09-30"]

ci = CausalImpact(signup_sales_mean, pre_period, post_period)
## 
ci.plot()

print(ci.summary())

print(ci.summary("report"))