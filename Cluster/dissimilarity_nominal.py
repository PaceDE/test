import pandas as pd
import numpy as np
pd.set_option('future.no_silent_downcasting', True)

data = {
    'id': ['1', '2', '3','4'],
    'testResult': ['CodeA', 'CodeB', 'CodeC','CodeA'],
    'FMarks': ['20', '30', '30','20']
}
df = pd.DataFrame(data)
print("\n Datasets")
print(df)
id=df["id"]
table = df.drop("id",axis=1)
n= len(table)
p= len(table.columns)
d_matrix=[[0.0 for j in range(n)] for i in range(n)]

def dissimilarity(row1,row2):
    m=np.sum((row1==row2))       
    return (p-m)/p


for i in range(n):
    for j in range(n):
        if i!=j:
            d_matrix[i][j]= dissimilarity(table.iloc[i],table.iloc[j])

print("\n Dissimilarity matrix")
print(" ","    ".join(id))
for i in range(n):
    print(id[i],end=" ")
    for j in range(n):
        print(f"{d_matrix[i][j]:.2f}",end=" ")
    print()
print("\nName: Dipesh Shrestha \nRoll no:08 \n")



        
