import pandas as pd
df = pd.read_csv('students.csv')

#Min_Max
Salary=df['Salary']
Xmin = Salary.min()
Xmax = Salary.max()
new_min = 0.0
new_max = 1.0
df['Min_Max_Norm'] = new_min + ((Salary - Xmin) / (Xmax - Xmin)) * (new_max - new_min)

#Z Score
df['Z_Score_Norm'] = (Salary-Salary.mean())/Salary.std()

#Decimal_Scaling
max_salary = max(Salary)
j = len(str(max_salary))
df['Decimal_Scaling_Norm'] = Salary/10**j

print(df)
print("\n Name: Dipesh Shrestha \n Roll no:08 \n")

df.to_csv('normalized_students.csv', index=False)