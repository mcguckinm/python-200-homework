import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


os.makedirs("outputs", exist_ok=True)


iris = load_iris(as_frame=True)

x = iris.data
y = iris.target

def print_header(title):
    print("\n" + "=" *70)
    print(title)
    print("=" * 70)



#preprocessing

print_header("Preprocessing")

#Q1

print("\nQuestion 1")

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

#Q2

print("\nQuestion 2")

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

print("Columns means of x_train_scaled: ")
print(x_train_scaled.mean(axis=0))
#we fit scaler on x train only so test does not leak information into preprocessing step

#KNN

print_header("KNN")

print("\nQuestion 1")

knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(x_train, y_train)
y_pred_knn_unscaled = knn_unscaled.predict(x_test)

print("Accuracy: ", accuracy_score(y_test, y_pred_knn_unscaled))

print("Classification Report: ")
print(classification_report(y_test, y_pred_knn_unscaled, target_names=iris.target_names))


#Q2

print("\nQuestion 2")

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(x_train_scaled, y_train)
y_pred_knn_scaled = (knn_scaled.predict(x_test_scaled))

print("Accuracy: ", accuracy_score(y_test, y_pred_knn_scaled))
# On Iris scaling might make little sifference because feature scales are already compared to real datasets

#Question 3

print("\nQuestion 3")

cv_scores_k5 = cross_val_score(
    KNeighborsClassifier(n_neighbors=5),
    x_train,
    y_train,
    cv=5
)

print("Fold scores: ", cv_scores_k5)
print("Mean CV score", cv_scores_k5.mean())
print("STD CV score: ", cv_scores_k5.std())

# cross validation is looking better overall more trust worthy than single train and test split
# overall averaging across multiple splots instead of one split makes data more reliable

#Question 4

print("\nQuestion 4")

k_values = [1, 3, 5, 7, 9, 11, 13, 15]
k_to_scores= {}

for k in k_values:
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), x_train, y_train, cv=5)
    mean_score = scores.mean()
    k_to_scores[k] = mean_score
    print(f"k={k:2d} mean cv score ={mean_score:4f}")

best_k = max(k_to_scores, key=k_to_scores.get)
print(f"Best k by mean CV score ={best_k}")

# I would choose k with the highest mean as it showed it performed best across validation folds

#classifier evaluation

print_header("Classifier Evaluation")

print("\nQuestion 1")

cm = confusion_matrix(y_test, y_pred_knn_unscaled)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("KNN Confusion Matrix (Unscaled Iris)")
plt.tight_layout()
plt.savefig("outputs/knn_confusion_matrix.png")
plt.close()

# if there is confusion, it is usually between versicolor and virginica.
#setosa is typically seprarated cleanly

#Decision Trees

print_header("Decision Trees")

print("Question 1")

tree = DecisionTreeClassifier(max_depth=3,random_state=42)
tree.fit(x_train, y_train)
y_pred_tree = tree.predict(x_test)

print("Accuracy: ", accuracy_score(y_test, y_pred_tree))
print("Classification Report:")
print(classification_report(y_test, y_pred_tree, target_names=iris.target_names))
# Compare this to accuracy model to see what did better on this split. 

#both models showed exact same Decision trees dont rely on distances so scaling usually doesnt change results

#Logistic Regression

print_header("Logistic Regression")

print("Question 1")

for c_value in [0.01, 1.0, 100]:
    log_reg = LogisticRegression(
        C=c_value,
        max_iter=1000,
        solver="liblinear"
    )
    log_reg.fit(x_train_scaled, y_train)
    total_coef_size = np.abs(log_reg.coef_).sum()
    print(f"C={c_value:<6} total coefficient magnitude = {total_coef_size:6f}")

# As C increases total coefficient magnitude increases which means regularization decreases.


#PCA

print_header("PCA")

digits = load_digits()
x_digits = digits.data
y_digits = digits.target
images = digits.images

def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using first n_components principal"""

    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction+ scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8,8)

#Question 1

print("\nQuestion 1")
print("x_digits shape: ", x_digits.shape)
print("images shape", images.shape)
fig, axes = plt.subplots(1,10, figsize=(15, 20))
for digit in range(10):
    idx = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[idx],cmap="gray_r")
    axes[digit].set_title(str(digit))
    axes[digit].axis("off")

plt.tight_layout()
plt.savefig("outputs/sample_digits.png")
plt.close()

#Question 2

print("\nQuestion 2")

pca = PCA()
pca.fit(x_digits)
scores = pca.transform(x_digits)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(scores[:,0], scores[:, 1], c=y_digits, cmap="tab10",s=10)
plt.colorbar(scatter, label="Digit")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Digits Dataset PCA Projection")
plt.tight_layout()
plt.savefig("outputs/pca_2d_projection.png")
plt.close()
# same-digit images tend to cluster together there is overlap because two components cannot capture all of the data.

#Q3

print("\nQuestion 3")
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_80 = np.argmax(cumulative_variance >=80)+1

plt.figure(figsize=(8,5))
plt.plot(range(1, len(cumulative_variance)+ 1), cumulative_variance, marker="o")
plt.axhline(0.80, linestyle="--")
plt.axvline(n_80, linestyle="--")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Digits PCA Cumulative Explained Variance")
plt.tight_layout()
plt.savefig("outputs/pca_variance_explained.png")
plt.close()

print("Approximate components needed for 80% variance: ", n_80)

#First component count where cumulative explained variance reaches or exceeds .80

#Question 4

print("\Question 4")

n_values = [2, 5, 15, 40]
sample_indices = [0, 1, 2, 3, 4]

fig, axes = plt.subplots(len(n_values) + 1, len(sample_indices), figsize=(10, 10))

for col, idx in enumerate(sample_indices):
    axes[0, col].imshow(images[idx], cmap="gray_r")
    if col ==0:
        axes[0, col].set_ylabel("Original", rotation=90, labelpad=20)
    axes[0,col].set_title(f"Digit {y_digits[idx]}")
    axes[0, col].axis("off")

for row, n_comp in enumerate(n_values, start=1):
    for col, idx in enumerate(sample_indices):
        reconstructed = reconstruct_digit(idx, scores, pca, n_comp)
        axes[row, col].imshow(reconstructed, cmap="gray_r")
        if col == 0:
            axes[row, col].set_ylabel(f"n={n_comp}", rotation=90, labelpad=20)
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png")
plt.close()

# digits usually become clearly recognizable by a moderate number of components, this lines up where the variance curve
# starts to level off.



