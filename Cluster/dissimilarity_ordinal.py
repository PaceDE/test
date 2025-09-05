import pandas as pd

import pandas as pd

data = {
    'id': ['1', '2', '3', '4'],
    'Satisfaction': ['Low', 'Medium', 'High', 'Medium'],
    'Education_Level': ['High School', 'College', 'College', 'Graduate']
}

df = pd.DataFrame(data)
print(f"\n{df}")

ids = df['id']
table = df.drop('id', axis=1)
n = len(table)
p = len(table.columns)

satisfaction_map = {'Low': 1, 'Medium': 2, 'High': 3}
education_map = {'High School': 1, 'College': 2, 'Graduate': 3}

df['Satisfaction'] = df['Satisfaction'].map(satisfaction_map)
df['Education_Level'] = df['Education_Level'].map(education_map)

m_s=len(df['Satisfaction'].unique())
m_e=len(df['Education_Level'].unique())

df['S_norm']=(df['Satisfaction']-1)/(m_s-1)
df['El_norm']=(df['Education_Level']-1)/(m_e-1)

def ordinal_dissimilarity(row1, row2):
    pass


d_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
def ordinal_dissimilarity(row1, row2):
    return abs(row1['S_norm'] - row2['S_norm']) + abs(row1['El_norm'] - row2['El_norm'])

for i in range(n):
    for j in range(n):
        if i != j:
            d_matrix[i][j] = ordinal_dissimilarity(df.iloc[i], df.iloc[j])

print("\t", "\t".join(ids))
for i in range(n):
    print(ids[i], end="\t")
    for j in range(n):
        print(f"{d_matrix[i][j]:.2f}", end="\t")
    print()
print("\nName: Dipesh Shrestha \nRoll no:08 \n")
