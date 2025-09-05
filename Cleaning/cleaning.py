import pandas as pd

data = pd.read_csv('student.csv')
print("\nBefore cleaning\n")
print(data)

#Fill missing age
mean_age=data['Age'].mean()
data['Age']=data['Age'].fillna(mean_age)

#Drop Duplicates
data.drop_duplicates(inplace=True)

#Standarize
gender_mapping={
    'M':'M', 
    'Male':'M', 
    'F':'F', 
    'Female':'F'
}
data['Gender'] = data['Gender'].map(gender_mapping)

#Removing Outliers
Q1 = data['Salary'].quantile(0.25)
Q3 = data['Salary'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['Salary'] >= Q1 - 1.5*IQR) & (data['Salary'] <= Q3 + 1.5*IQR)]

print("\n After Cleaning \n")
print(data)
data.to_csv("students.csv",index=False)


print("\n Name: Dipesh Shrestha \n Roll no:08 \n")
