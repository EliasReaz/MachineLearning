##############################
# Association Rule - Apriori
##############################

from apyori import apriori

import pandas as pd
import numpy as np

## import data
data = pd.read_csv("data/sample_data_apriori.csv")
print("shape of data: ", data.shape)
print(data.head())
print("info:")
print("-------------")
print(data.info())

## drop transaction_id

data.drop(["transaction_id"], axis=1, inplace=True)

data["product1"].nunique()

## modify data for apriori algorithm

transaction_list = []

for index, row in data.iterrows():
    transaction = list(row.dropna())
    transaction = [w.strip() for w in transaction]
    transaction = [w.replace("America ", "American") for w in transaction]
    transaction_list.append(transaction)
    
###################################################
# Apply apriori algorithm
###################################################

# apriori_rule is a generator.
apriori_rule = apriori(transaction_list, 
                       min_support = 0.0003,
                       min_confidence= 0.20,
                       min_lift=3,
                       min_length=2,
                       max_length =2)
## Note if we dontot set min_length and max_length, it would
## take a very very long time to run.

# convert genrator to a list

apriori_rule_list = list(apriori_rule)

print(apriori_rule_list[0])

##############################################
# convert to a dataframe
############################################### 

len(apriori_rule_list[0])
apriori_rule_list[0][2][0][0]

product1 = [list(rule[2][0][0])[0] for rule in apriori_rule_list]
product2 = [list(rule[2][0][1])[0] for rule in apriori_rule_list]
support = [round(rule[1],4) for rule in apriori_rule_list]
confidence = [round(rule[2][0][2], 4) for rule in apriori_rule_list]
lift = [round(rule[2][0][3],4) for rule in apriori_rule_list]

apriori_rule_df = pd.DataFrame({"product1":product1,
                                "product2": product2,
                                "support":support,
                                "confidence": confidence,
                                "lift": lift})

apriori_rule_df.sort_values(by="lift", ascending=False, inplace=True)





 