import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

os.makedirs("outputs",exist_ok=True)

#scikit Lean API

#Q1
print("\nscikit-learn API Q1")

years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)

pred_4 = model.predict([[4]])[0]
pred_8 = model.predict([[8]])[0]

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted salary for 4 years:", pred_4)
print("Predicted salary for 8 years:", pred_8)

#Q2

print("\nscikit-learn API Q2")

x = np.array([10, 20, 30, 40, 50])
print("Original shape: ", x.shape)


x_2d = x.reshape(-1, 1)
print("Reshaped shape: ", x_2d.shape)


#this is a one dimension array which is what is expected when I run it. There is only 1 row and 5 columns. Which is exactly what it is saying.
# when we reshaped it we see it show from rows to columns. still need stucture of it. 


#Q3

print("\nscikit-learn Q3")

x_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(x_clusters)
labels = kmeans.predict(x_clusters)

print("Cluster centers:")
print(kmeans.cluster_centers_)
print("Points in each cluster:")
print(np.bincount(labels))

plt.figure()
plt.scatter(x_clusters[:, 0], x_clusters[:, 1], c=labels)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker = "X",
    s = 200,
    c="black"
)
plt.title("KMeans Cluster")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("outputs/kmeans_clusters.png")
plt.close()

# Linear Regression

np.random.seed(42)
num_patients = 100
age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)

cost = 200 * age +15000 * smoker + np.random.normal(0, 3000, num_patients)

#Q1

print("Linear Regression Q1")

plt.figure()
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical cost vs Age")
plt.xlabel("Age")
plt.ylabel("Medical Cost")
plt.savefig("outputs/cost_vs_age.png")
plt.close()

# Blue shows lower cost for non smoking people and red shows for smoking high cost individuals. 
#suggests smoking increases cost


#Q2

print("Linear Regression Q2")

x_age = age.reshape(-1, 1)
y = cost

x_train, x_test, y_train, y_test = train_test_split(
    x_age, y, test_size=0.2, random_state=42
)

print("X_train shape:", x_train.shape)
print("X_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#Q3

print("\nQuestion 3")

model_age = LinearRegression()
model_age.fit(x_train, y_train)

y_pred = model_age.predict(x_test)

rmse_age = np.sqrt(np.mean((y_pred - y_test)** 2))
r2_age = model_age.score(x_test, y_test)

print("Slope: ",model_age.coef_[0])
print("Intercept: ", model_age.intercept_)
print("RMSE:", rmse_age)
print("R^2 on test set: ",r2_age)


#Slope tells me how much cost has changed for each additional year of age assuming in hundreds of dollars fiven the #
# current results it shows about 196 increase

#Q4
print("\nQuestion 4")

x_full = np.column_stack([age, smoker])

x_train_full, x_test_full, y_train_full, y_test_full = train_test_split(
    x_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(x_train_full, y_train_full)
y_pred_full = model_full.predict(x_test_full)

r2_full = model_full.score(x_test_full, y_test_full)

print("R^2 using age only: ",r2_age)
print("R^2 using age + smoker: ", r2_full)
print("Age coefficient: ", model_full.coef_[0])
print("Smoker Coefficient: ", model_full.coef_[1])

#smoker coefficient tellsus how much more it will cost by prediction. This model shows us that it is a significant cost
#This shows us over the non smokers while keeping age consistent.

#Q5

print("\nLinear Regression Q5")

plt.figure()
plt.scatter(y_pred_full, y_test_full)

min_val = min(y_pred_full.min(), y_test_full.min())
max_val = max(y_pred_full.max(), y_test_full.max())

plt.plot([min_val, max_val], [min_val, max_val], "k--")

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()


#Linear regression line anything above means it under predicted while a point below means it over predicted which is
#the case in this that we over predicted the costs completely. 