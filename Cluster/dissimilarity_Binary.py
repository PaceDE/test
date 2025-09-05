import pandas as pd
import numpy as np
pd.set_option('future.no_silent_downcasting', True)


data = {
    'Name': ['Jack', 'Mary', 'Jim'],
    'Gender': ['M', 'F', 'M'],
    'Fever': ['Y', 'Y', 'Y'],
    'Cough': ['N', 'N', 'P'],
    'Test-1': ['P', 'P', 'N'],
    'Test-2': ['N', 'N', 'N'],
    'Test-3': ['N', 'P', 'N'],
    'Test-4': ['N', 'N', 'N'],
}
df = pd.DataFrame(data)

print("\n Datasets")
print(df)
names=df["Name"]
table = df.drop(columns=["Gender","Name"])
mapping= {'Y':1,'P':1,'N':0}
table=table.replace(mapping)
n= len(table)
d_matrix=[[0.0 for j in range(n)] for i in range(n)]

def dissimilarity(row1,row2):
    a=np.sum((row1==1)&(row2==1))
    b=np.sum((row1==1)&(row2==0))
    c=np.sum((row1==0)&(row2==1))

    return (b+c)/(a+b+c) if (a+b+c) >0 else 0


for i in range(n):
    for j in range(n):
        if i!=j:
            d_matrix[i][j]= dissimilarity(table.iloc[i],table.iloc[j])

print("\n Disimilarity matrix")
print("\t"," ".join(names))
for i in range(n):
    print(names[i],end="\t")
    for j in range(n):
        print(f"{d_matrix[i][j]:.2f}",end=" ")
    print()

print("\nName: Dipesh Shrestha \nRoll no:08 \n")



        
