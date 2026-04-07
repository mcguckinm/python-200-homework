import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


os.makedirs("outputs", exist_ok=True)

# Task 1:

df = pd.read_csv("../assignments/resources/student_performance_math.csv", sep=";")

print("\nTask 1")

print("Shape: ", df.shape)
print("\nFirst five rows: ")
print(df.head())
print("\nDataTypes: ")
print(df.dtypes)


plt.figure()
plt.hist(df["G3"], bins=np.arange(-0.5, 21.5, 1), edgecolor="black")
plt.title("Distribution of Final Math Grades")
plt.xlabel("G3")
plt.ylabel("Count")
plt.savefig("outputs/g3_distribution.png")
plt.close()

#Task 2

print("\nTask 2")

print("Original shape: ", df.shape)
df_clean = df[df["G3"] != 0].copy()

print("Filtered shape: ",df_clean.shape)
print("Rows removed: ", len(df) - len(df_clean))

#Removing G3=0 rows as those represent students who did not take the exam, not the ones who actually earned a 0.
#Keeping them gives a false assumption of data about academic performance that would be incorrect

yes_no_cols = ["schoolsup", "internet", "higher", "activities"]

for col in yes_no_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})

df["sex"] = df["sex"].map({"F": 0, "M": 1})
df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})

corr_original = df["absences"].corr(df["G3"])
corr_filtered = df_clean["absences"].corr(df_clean["G3"])

print("Correlation absences vs G3 (original): ", corr_original)
print("Correlation absences vs G3 (filtered): ", corr_filtered)

# Filtering data changes the amount of issues we have in G3=0 rows. Students who were absent for the final exam having them mixed will distort
#the final number with making sure you have students who did or did not take the exam. 

#task 3
print("\nTask 3")

numeric_cols = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "absences", "freetime", "goout", "Walc",
    "schoolsup", "internet", "higher", "activities", "sex"
]


correlations = df_clean[numeric_cols + ["G3"]].corr()["G3"].drop("G3").sort_values()
print("Correlations with G3:")
print(correlations)

plt.figure()
plt.scatter(df_clean["failures"], df_clean["G3"], alpha=0.7)
plt.title("Failures vs Final Grade")
plt.xlabel("Failures")
plt.ylabel("Final Grade")
plt.savefig("outputs/failures_vs_g3.png")
plt.close()

#Plot should show that more past failues are associated with lower grades

plt.figure()
plt.scatter(df_clean["studytime"], df_clean["G3"], alpha=0.7)
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("G3")
plt.savefig("outputs/studytime_vs_g3.png")
plt.close()

#Plot should show that more study time is associated with higher grades

#Task 4

print("\nTask 4")

x_base = df_clean[["failures"]].values
y_base = df_clean["G3"].values

x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(x_base, y_base, test_size=0.2, random_state=42)

model_base = LinearRegression()
model_base.fit(x_train_b, y_train_b)

y_pred_b = model_base.predict(x_test_b)

rmse_base = np.sqrt(np.mean((y_test_b - y_pred_b) ** 2))
r2_base = model_base.score(x_test_b, y_test_b)

print("Slope: ", model_base.coef_[0])
print("RMSE: ", rmse_base)
print("R^2: ", r2_base)


#Negative slope indicates failures as this is predicted final grade changes
#RMSE shows 2.96 which means on average our predictions are off by about 3 grade points
#R^2 shows us 0.089 which means about 8.9% of the variance in final grades can be explained by the number of past failures

#task 5

print("\nTask 5")

feature_cols = [
    "failures", "Medu", "Fedu", "studytime", "higher", "schoolsup",
    "internet", "sex", "freetime", "activities", "traveltime"
]

x = df_clean[feature_cols].values
y = df_clean["G3"].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

train_r2 = model.score(x_train, y_train)
test_r2 = model.score(x_test, y_test)
rmse_full = np.sqrt(np.mean((y_pred - y_test) ** 2))


print("Baseline test R^2: ", r2_base)
print("Full model train R^2: ", train_r2)
print("Full model test R^2: ", test_r2)
print("Full model RMSE: ", rmse_full)


print("\nCoefficients")

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Look for largest positive and negative coefficients after running
# looking at data only largest significant change is schoolsup at -2.062
#since R^2 train and R^2 test are currently close in multiple factions this tells me 
# it is generalizing well, larger gaps would suggest overfitting.

# in production i would keep features that are meaningful, available early, and stable, and consider dropping weak or hard to interpret. 


#task 6

print("\nTask 6")

plt.figure()
plt.scatter(y_pred, y_test, alpha = .07)

min_val = min(y_pred.min(), y_test.min())
max_val = max(y_pred.max(), y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], "k--")
plt.title("Predicted vs Actual (full model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()

#points above the line would indicate true grade is higher than predicted
#poiints below means model is over-predicted

print("Filtered dataset size: ", df_clean.shape)
print("Test set size: ", x_test.shape[0])
print("Best model RMSE: ", rmse_full)
print("Best model R^2: ", test_r2)

coef_series = pd.Series(model.coef_, index=feature_cols).sort_values()
print("\nMost negative coefficients: ")
print(coef_series.head(2))
print("\nMost positive coefficients: ")
print(coef_series.tail(2))

#filtered dataset excludes students who missed the final exam
#RMSE shows the model's typical preduction error in grade points on a 0-20 scale
#R^2 shows how much variation in final grade the model explains
# The biggest positive coefficients indicate features associated with higher predicted grades
# biggest negative indicates lower predicted grades
# one suprising result is a feature that differs from what you expect from intuition or raw correlations


print("\n Neglected Feature: The power of G1")

feature_cols_g1 = feature_cols + ["G1"]
x_g1 = df_clean[feature_cols_g1].values
y_g1 = df_clean["G3"].values

x_train_g1, x_test_g1, y_train_g1, y_test_g1 = train_test_split(
    x_g1, y_g1, test_size=0.2, random_state=42
)

model_g1 = LinearRegression()
model_g1.fit(x_train_g1, y_train_g1)

test_r2_g1 = model_g1.score(x_test_g1, y_test_g1)
print("Test R^2 with G1 added: ", test_r2_g1)

# A high R^2 in G1 does not mean G1 causes G3. It means strong predictors as it captures earlier performance in class
#This is useful for forcasting later outcomes, but is overall less useful for early intervention.
