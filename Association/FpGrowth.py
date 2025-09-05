import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules # type: ignore

np.seterr(invalid="ignore")

transactions = [
    ['M', 'O', 'N','K','E','Y'],
    ['D', 'O', 'N','K','E','Y'],
    ['M', 'A','K','E'],
    ['M', 'U', 'C','K','Y'],
    ['C', 'O', 'O','K','I','E'],
]
te = TransactionEncoder()
te_matrix = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_matrix, columns=te.columns_)

frequent_itemsets = fpgrowth(df, min_support=0.6, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence']])
print("\n Name: Dipesh Shrestha \n Roll no:08 \n")
