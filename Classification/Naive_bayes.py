import pandas as pd
import numpy as np

data = pd.read_csv("Naive.csv")

features=['Age','Income','Student','Credit_Rating']
X= data.drop('Buys_Computer',axis=1)
Y = data['Buys_Computer']
classes = Y.unique()

priors={}

def prior_probability():
    for cls in classes:
        priors[cls]= len(Y[Y==cls])/len(Y)


def likelihood(feature_values):
    likelihood={}
    categories=feature_values.unique()
    for cls in classes:
        likelihood[cls] ={}
        
        class_count=len(Y[Y==cls])
        for cat in categories:
            count=np.sum((feature_values==cat) & (Y==cls))
            likelihood[cls][cat]=(count+1)/ (class_count+len(categories))
    return likelihood   

def predict(sample):
    posterior_probab={}
    for cls in classes:
        posterior_probab[cls]=priors[cls]
        posterior_probab[cls]*=age_likelihood[cls].get(sample["Age"], 1e-6)
        posterior_probab[cls]*=income_likelihood[cls].get(sample["Income"], 1e-6)
        posterior_probab[cls]*=student_likelihood[cls].get(sample["Student"], 1e-6)
        posterior_probab[cls]*=cr_likelihood[cls].get(sample["Credit_Rating"], 1e-6)
    predicted_class=max(posterior_probab, key=posterior_probab.get)
    return(posterior_probab,predicted_class)

prior_probability()

age_likelihood= likelihood(X["Age"])
income_likelihood= likelihood(X["Income"])
student_likelihood= likelihood(X["Student"])
cr_likelihood= likelihood(X["Credit_Rating"])

sample = {"Age":"youth","Income":"medium","Student":"yes","Credit_Rating":"fair"}
posterior, pred_class = predict(sample)

print(posterior)
print(f"Predicted Class for Age: {sample['Age']}, Income: {sample['Income']}, Student: {sample['Student']}, Credit_Rating: {sample['Credit_Rating']} -> {pred_class}")

print("\n Name: Dipesh Shrestha \n Roll no:08 \n")




