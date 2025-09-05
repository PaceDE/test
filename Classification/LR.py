import pandas as pd

df = pd.read_excel("LR.xlsx")

x_mean = df['Experience'].mean()
y_mean = df['Salary'].mean()

df["xi-xmean"] =df['Experience'] - x_mean
df["yi-ymean"]= df['Salary'] - y_mean
df["(xi-xmean) * (yi-ymean)"]= df["xi-xmean"]*df["yi-ymean"]
df["(xi-xmean)^2"]=df["xi-xmean"]**2

numerator = df["(xi-xmean) * (yi-ymean)"].sum()
denominator = df["(xi-xmean)^2"].sum()

m = numerator / denominator

c = y_mean - m * x_mean

df["Predicted"] = df['Experience'].apply(lambda x: m * x + c)
predicted_value = m * 10 + c

print()
print(df)

print(f"\nx̄ (x mean): {x_mean}")
print(f"ȳ (y mean): {y_mean}")
print(f"\nSlope (m): {m}")
print(f"Intercept (c): {c}")
print(f"Linear regression model y = {m:2f}x + {c:2f}")
print(f"Predicted Salary for 10 years Experience : {predicted_value}")

print("\nName: Dipesh Shrestha \nRoll no:08 \n")
