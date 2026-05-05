import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


#===================

# setup

#===================
os.makedirs("outputs", exist_ok=True)


def print_section(title: str) -> None:
    print("\n"+ "=" * 80)
    print("\n"+title)
    print("\n" + "=" * 80)

def summarize_binary_target(target_series: pd.Series) -> None:
    class_counts = target_series.value_counts().sort_index()
    class_props = target_series.value_counts(normalize=True).sort_index()

    print("Class counts: ")
    print(class_counts)
    print("\nClass proportions: ")
    print(class_props)

    baseline_accuracy = class_props.max()
    print(f"\n Baseline accuracy (always predict majority class): {baseline_accuracy:.4f}")

    print("\nInterpretation")
    print(
        f"A model must beat {baseline_accuracy:.4f} to be meaningful better than a trivial baseline"
    )
    print(
        "- If a model is only slightly above baseline, it may not be learning much beyond class imbalance"
    )

def interpret_accuracy(model_name: str, accuracy: float, baseline: float) -> None:
    print(f"\nInterpretation for {model_name}:")
    print(f" - Accuracy = {accuracy:.4f}")
    print(f"- Baseline = {baseline:.4f}")

    if accuracy >= 0.95:
        strength = "very strong"
    elif accuracy >= 0.90:
        strength = "strong"
    elif accuracy >= 0.80:
        strength = "moderate"
    else:
        strength = "weak"

    print(f"- This is a {strength} result based on accuracy alone.")

    improvement = accuracy - baseline
    print(f"- Improvement over Baseline: {improvement:.4f}")

    if improvement < 0.03:
        print("- This model improves slightly over baseline, sp ots pratical value may be limited.")
    elif improvement < 0.10:
        print("- This model improves meaningfully over baseline, but there is still room to improve.")
    else:
        print("- This mode improves clearly over the baseline and is learning useful patterns.")


    print("- In spam filtering, accuracy is not enough by itself")
    print("- False positives matter as some real emails may get flagged as spam.")
    print("- False negatives matter too because spam may be getting through.")
    print("- A good next step would be comparing precision and recall for the spam class.")

def interpret_confusion_matrix(cm: np.ndarray) -> None:
    tn, fp, fn, tp = cm.ravel()
    
    print("\nConfusion matrix interpretation:")
    print(f"- True Negatives (ham predicted as ham): {tn}")
    print(f"False Positives (ham predicted as spam): {fp}")
    print(f"- False Negatives (spam predicted as ham): {fn}")
    print(f"- True Positives (spam predicted as spam): {tp}")

    if fp > fn:
        print("- This model makes more false positive predictions than false negatives.")
        print("- That means it is more aggressive and may block legitimate email too often.")
    elif fn > fp:
        print("-This model makes more false negatives than false positives.")
        print("- This means it lets more spam through than it blocks incorrectly.")
    else:
        print("- False positives and false negatives are balanced in this test split.")

    print("- In many real email systems, false positives are often considered too costly.")

def print_top_feature_importance(model, feature_names, top_n=10, label="Model"):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

    print(f"\nTop {top_n} features for {label}:")
    print(importances.head(top_n).round(4))

    print(f"\nInterpretation for {label}:")
    print("- These are variables the model relied on most when making splits or ensemble decisions.")
    print("- Higher importance means the feature helped separate spam from ham more often.")
    print("- Low-importance features might add little signale and could be candidates for removal or further review.")

    return importances

def interpret_cross_validation(model_name: str, mean_score: float, std_score: float) -> None:
    print(f"\nCross-validation interpretation for {model_name}")
    print(f"- Mean CV accuracy: {mean_score:.4f}")
    print(f"Std CV accuracy: {std_score:.4f}")

    if std_score < 0.01:
        print("- The model is very stable across folds.")
    elif std_score < 0.03:
        print("- Model is reasonably stable across folds")
    else:
        print("- Model shows noticeable variability across folds.")
    
    print("- Cross-validation is more trustworthy than a single test split because it averages over several folds.")



#====================

#Task 1

#====================

print_section("Task 1: Load and Explore Data")

spam_data = fetch_openml(name="spambase", version=1, as_frame=True)
spam_df = spam_data.frame.copy()

target_column = spam_data.target.name if hasattr(spam_data.target, "name") and spam_data.target.name else spam_df.columns[-1]

spam_df[target_column] = spam_df[target_column].astype(int)

feature_names = [col for col in spam_df.columns if col != target_column]

x_features = spam_df.drop(columns=[target_column])
y_target = spam_df[target_column]

print(f"Dataset shape: {spam_df.shape}")
print(f"Number of emails: {len(spam_df)}")
summarize_binary_target(y_target)

print("\nFirst 10 feature names:")
print(feature_names[:10])

baseline_accuracy = y_target.value_counts(normalize=True).max()

word_free_col = "word_freq_free"
capital_total_col = "capital_run_length_total"

# Looking for exclamation mark frequency column

char_frequency_columns = [col for col in spam_df.columns if "char_freq" in col]
print("\nCharacter frequency columns")
print(char_frequency_columns)

char_exclamation_col = None

# First try exact common name

if "char_freq_!" in spam_df.columns:
    char_exclamation_col = "char_freq_!"
elif "char_freq_%21" in spam_df.columns:
    char_exclamation_col = "char_freq_%21"
else:
    #Fall back: Look for char_freq containing '!'
    for col in spam_df.columns:
        if "%21" in col or "!" in col:
            char_exclamation_col = col
            break
# If still not found use a fallback to explain what happened

if char_exclamation_col is None:
    print("\nCould not find an exclamation mark column exactly as expected.")
    print("Using first available char_freq column instead for plotting")
    if len(char_exclamation_col) > 0:
        char_exclamation_col = char_frequency_columns[0]
    else:
        raise ValueError("No char_freq columns were found in the dataset.")
for expected_col in [word_free_col, char_exclamation_col, capital_total_col]:
    if expected_col not in spam_df.columns:
        raise ValueError(
            f"Expected column '{expected_col}' not found."
            f"Available columns {list(spam_df.columns[:20])}"
        )
print(f"\nUsing exlamation-like column: {char_exclamation_col}")


for column_name in [word_free_col, char_exclamation_col, capital_total_col]:
    plt.figure(figsize=(8,5))
    plt.boxplot([
        spam_df.loc[spam_df[target_column] == 0, column_name],
        spam_df.loc[spam_df[target_column] == 1, column_name]
    ],
    tick_labels=["Ham (0)", "Spam (1)"]
    )
    plt.title(f"{column_name}: Spam vs Ham")
    plt.ylabel(column_name)
    plt.tight_layout()

    output_name = f"outputs/{column_name}_boxplot.png".replace("!", "exclamation")
    plt.savefig(output_name)
    plt.close()
    print(f"Saved {output_name}")

print("\nZero-heavy feature check:")
for column_name in [word_free_col, char_exclamation_col]:
    zero_rate = (spam_df[column_name] == 0).mean()
    print(f"- {column_name}: zero proportion = {zero_rate:.4f}")

print("\nFeature scale summary (first 15 rows)")
print(x_features.describe().loc[["mean", "std", "min", "max"]].T.head(15).round(4))

print("\nInterpretation:")
print("- Many word-frequency features are heavily zero-inflated, meaning most emails do not contain many tracked words")
print("- Some features operate on different scales")
print("- That matters because KNN, Logistic Regression, and PCA are sensitive to feature magnitudes.")
print("- This is why scaling is important before using those methods")

#============================

#Task 2

#============================

print_section("Task 2: Prepare the Data")

x_train, x_test, y_train, y_test = train_test_split(
    x_features,
    y_target,
    test_size=0.20,
    stratify=y_target,
    random_state=42
)

print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


feature_scaler = StandardScaler()

x_train_scaled = feature_scaler.fit_transform(x_train)
x_test_scaled = feature_scaler.fit_transform(x_test)

print("\nScaling check:")
print(f"- Any NaN in x_train_scaled? {np.isnan(x_train_scaled).any()}")
print(f"- Any inf in x_train_scaled? {np.isinf(x_train_scaled).any()}")
print(f"- Max absolute value in x_train_scaled: {np.abs(x_train_scaled).max():.4f}")

pca_model = PCA(svd_solver="full")
pca_model.fit(x_train_scaled)


cumulative_variance = np.cumsum(pca_model.explained_variance_ratio_)
components_for_90 = np.argmax(cumulative_variance >= 0.90) + 1


plt.figure(figsize=(8,5))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker="o")
plt.axhline(0.90, linestyle="--")
plt.axvline(components_for_90, linestyle="--")
plt.xlabel("Number of Components")
plt.ylabel("Cumlative Explained Variance")
plt.tight_layout()
plt.savefig("outputs/pca_variance_90_spambase.png")
plt.close()

print("Saved ouputs/pca_variance_90_spambase.png")
print(f"Components needed for 90% explained variance: {components_for_90}")

x_train_pca = pca_model.transform(x_train_scaled)[:, :components_for_90]
x_test_pca = pca_model.transform(x_test_scaled)[:, :components_for_90]

print("\nInterpretation:")
print("- PCA reduces dimensionality by combining original variables into principal components.")
print("- Retaining 90% variance keeps most of the information while reducing the number of dimensions.")
print("- PCA may help KNN and Logistic Regression by reducing redundancy and noise.")
print("- PCA is usually less useful for tree-based models because trees can handle raw features.")

#=========================

#Task 3

#==========================


print_section("Task 3: Compare Classifiers")

model_results = {}

#KNN on unscaled data

print_section("Task 3A: KNN on Unscaled Data")

knn_unscaled_model = KNeighborsClassifier(n_neighbors=5)
knn_unscaled_model.fit(x_train, y_train)
knn_unscaled_predictions = knn_unscaled_model.predict(x_test)

knn_unscaled_accuracy = accuracy_score(y_test, knn_unscaled_predictions)
model_results["KNN unscaled"] = knn_unscaled_accuracy

print(f"Accuracy: {knn_unscaled_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test,knn_unscaled_predictions))
interpret_accuracy("KNN unscaled", knn_unscaled_accuracy, baseline_accuracy)

print("\nWhy this result may happen?")
print("- KNN depends on distance between points.")
print("- When features are on different scales, large-scale variables can dominate distance calculations.")
print("- That often hurts KNN performance on unscaled data.")

#KNN on scaled data

print_section("Task 3B: KNN on Scaled Data")


knn_scaled_model = KNeighborsClassifier(n_neighbors=5)
knn_scaled_model.fit(x_train_scaled, y_train)
knn_scaled_predictions = knn_scaled_model.predict(x_test_scaled)

knn_scaled_accuracy = accuracy_score(y_test, knn_scaled_predictions)
model_results["KNN scaled"] = knn_scaled_accuracy

print(f"Accuracy: {knn_scaled_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, knn_scaled_predictions))
interpret_accuracy("KNN scaled", knn_scaled_accuracy, baseline_accuracy)

print("\nWhy did this happen?")
print("- Scaling gives each feature a more equal role in distance calculations.")
print("- If scaled KNN improves over unscaled KNN, that suggests feature scale matters.")

# KNN on PCA data

print_section("Task 3C: KNN on Scaled + PCA Data")

knn_pca_model = KNeighborsClassifier(n_neighbors=5)
knn_pca_model.fit(x_train_pca, y_train)
knn_pca_predictions = knn_pca_model.predict(x_test_pca)

knn_pca_accuracy = accuracy_score(y_test, knn_pca_predictions)
model_results["KNN PCA"] = knn_pca_accuracy

print(f"Accuracy: {knn_pca_accuracy:.4f}")
print("\nClassification Report")
print(classification_report(y_test, knn_pca_predictions))
interpret_accuracy("KNN PCA", knn_pca_accuracy, baseline_accuracy)

print("\nWhy this result may happen?")
print("- PCA can remove noise and redundant dimensions.")
print("- But PCA can also compress away details that help classification.")
print("- If PCA helps, it suggest original space had redundancy and noise.")
print("- If PCA hurts, it suggests some useful information was lost.")

# Decision Tree depth comparison 

print_section("Task 3D: Decision Tree Depth Comparison")

depth_candidates = [3, 5, 10, None]
depth_performance = {}

for max_depth_value in depth_candidates:
    depth_tree_model = DecisionTreeClassifier(max_depth=max_depth_value, random_state=42)
    depth_tree_model.fit(x_train, y_train)

    train_accuracy = depth_tree_model.score(x_train, y_train)
    test_accuracy = depth_tree_model.score(x_test, y_test)

    depth_performance[max_depth_value] = (train_accuracy, test_accuracy)

    print(
        f"max_depth={str(max_depth_value):>4} |"
        f"train_accuracy={train_accuracy:.4f} |"
        f"test_accuracy={test_accuracy:.4f}"
    )

print("\nInterpretation:")
print("- If training accuracy rises much faster than test accuracy, the tree is likely overfitting.")
print("- A shallower tree is often easier to explain and generalizes better.")
print("- The best depth is not always the deepest one.")

# Chosen tree depth

chosen_tree_depth = max(
    depth_performance,
    key=lambda depth: depth_performance[depth][1] - abs(depth_performance[depth][0] - depth_performance[depth][1]) * 0.2
)

print(f"\nChosen tree depth: {chosen_tree_depth}")

deicsion_tree_model = DecisionTreeClassifier(max_depth=chosen_tree_depth, random_state=42)
deicsion_tree_model.fit(x_train, y_train)
decision_tree_predictions = deicsion_tree_model.predict(x_test)

decision_tree_accuracy = accuracy_score(y_test, decision_tree_predictions)
model_results["Decision Tree"] = decision_tree_accuracy

print(f"Accuracy: {decision_tree_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, decision_tree_predictions))
interpret_accuracy("Decision Tree", decision_tree_accuracy, baseline_accuracy)

print("\nWhy this result happend?")
print("- Trees are not sensitive to feature scaling.")
print("- They can capture nonlinear rules and interactions.")
print("- But deep trees can memorize training data and overfit.")

# Random Forest

print_section("Task 3E: Random Forest")

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(x_train, y_train)
random_forest_predictions = random_forest_model.predict(x_test)

random_forest_accuracy = accuracy_score(y_test, random_forest_predictions)
model_results["Random Forest"] = random_forest_accuracy

print(f"Accuracy: {random_forest_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, random_forest_predictions))
interpret_accuracy("Random Forest", random_forest_accuracy, baseline_accuracy)

print("\nWhy this result may happen?")
print("- Random forest reduce overfitting by averaging many trees.")
print("- They usually perform better than a single tree when there are many useful predictors.")
print("- Strong performance here suggests the dataset has learnable nonlinear structure.")

# Logistic Regressionon scaled data

print_section("Task 3F: Logistic Regression on Scaled Data")

logistic_scaled_model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
logistic_scaled_model.fit(x_train_scaled, y_train)
logistic_scaled_predictions = logistic_scaled_model.predict(x_test_scaled)

logistic_scaled_accuracy = accuracy_score(y_test, logistic_scaled_predictions)
model_results["Logistic Regression scaled"] = logistic_scaled_accuracy

print(f"Accuracy: {logistic_scaled_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test,logistic_scaled_predictions))
interpret_accuracy("Logistic Regression scaled", logistic_scaled_accuracy, baseline_accuracy)


print("\nWhy this happened:")
print("- Logistic Regression works well when the classes can be separated linearly.")
print("- Scaling helps because the optimization procedure is sensitive to magnitude.")
print("- If logistic Regression performs well, that suggest many useful signals combine in a smooth way.")

#Logistic Regression on PCA data

print_section("Task 3G: Logistic Regression on scaled + PCA Data")

logistic_pca_model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
logistic_pca_model.fit(x_train_pca, y_train)
logistic_pca_predications = logistic_pca_model.predict(x_test_pca)

logistic_pca_accuracy = accuracy_score(y_test, logistic_pca_predications)
model_results["Logistic Regression PCA"] = logistic_pca_accuracy

print(f"Accuracy: {logistic_pca_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, logistic_pca_predications))
interpret_accuracy("Logistic Regression PCA", logistic_pca_accuracy,baseline_accuracy)

print("\nWhy this happened:")
print("- PCA may remove noise before Logisitic Regression fits the decision boundary")
print("- PCA may also combine features in ways that slightly weaken class separation.")
print("- Comparing scaled Logistic Regression vs PCA Logistic Regression tells you whether dimensionality reduction helped.")


# Summary of Models

print_section("Task 3H: Model Summary")

for model_name, model_accuracy in sorted(model_results.items(), key=lambda item: item[1], reverse=True):
    print(f"{model_name:<30} {model_accuracy:.4f}")

best_model_name = max(model_results, key=model_results.get)
print(f"\nBest model by test accuracy: {best_model_name}")

print("\nOverall Interpretation:")
print("- The best model is one that achieved the highest test accuracy on this split.")
print("- But the 'best' model should also be judged by stability, false positive behavior, and interpretability.")
print("- For spam filtering, blindly maximizing accuracy is not enough.")
print("- The real question is whether the model handles the cost of false positives and false negatives appropriately.")

# ===================

# Task 3I confusion matrix for best model

#=====================

print_section("Task 3I: Confusion Matrix for Best Model")

predictions_by_models = {
    "KNN unscaled": knn_unscaled_predictions,
    "KNN scaled": knn_scaled_predictions,
    "KNN PCA": knn_pca_predictions,
    "Decision Tree": decision_tree_predictions,
    "Random Forest": random_forest_predictions,
    "Logistic Regression scaled": logistic_scaled_predictions,
    "Logistic Regression PCA": logistic_pca_predications,
}

best_predictions = predictions_by_models[best_model_name]
best_confusion_matrix = confusion_matrix(y_test, best_predictions)

confusion_display = ConfusionMatrixDisplay(
    confusion_matrix=best_confusion_matrix,
    display_labels=["Ham (0)", "Spam (1)"]
)
fig, ax = plt.subplots(figsize=(6,5))
confusion_display.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title(f"Best Model Confusion Matrix {best_model_name}")
plt.tight_layout()
plt.savefig("outputs/best_model_confusion_matrix.png")
plt.close()

interpret_confusion_matrix(best_confusion_matrix)

#===================

# Task 3J Feature importance

#====================

print_section("Task 3J: Feature Importance")

decision_tree_importance = print_top_feature_importance(
    deicsion_tree_model,
    x_features.columns,
    top_n=10,
    label="Decision Tree"
)

random_forest_importance = print_top_feature_importance(
    random_forest_model,
    x_features.columns,
    top_n=10,
    label="Random Forest"
)

top_random_forest_features = random_forest_importance.head(10).sort_values()

plt.figure(figsize=(10,6))
plt.barh(top_random_forest_features.index, top_random_forest_features.values)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 10 Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("outputs/feature_importances.png")
plt.close()

print("\nWhat to learn from these features:")
print("- The most important features are the strongest signals for distinguishing spam from ham.")
print("- If the same features appear near the top in both the tree and the forest, that increases confidence they are useful.")
print("- Features near zero importance may be mostly noise or redundant.")
print("- A next step could be testing whether a reduced-feature model performs nearly as well.")

#==========================

#Task 4: Cross-validation

#==========================

print_section("Task 4: Cross-Validation")

cross_vallidation_models = {
    "KNN unscaled": (
        KNeighborsClassifier(n_neighbors=5),
        x_train
    ),
    
    "KNN scaled": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),
        x_train
    ),

    "KNN PCA": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=components_for_90, svd_solver="full")),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),
        x_train
    ),

    "Decision Tree": (
        DecisionTreeClassifier(max_depth=chosen_tree_depth, random_state=42),
        x_train
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=100, random_state=42),
        x_train
    ),

    "Logistic Regression scaled": (
        Pipeline([
            ("scaled", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))
        ]),
        x_train
    ),

    "Logistic Regression PCA": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=components_for_90, svd_solver="full")),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))
        ]),
        x_train
    ),
}

cross_validation_summary = {}

for model_name, (model_object, x_cv_data) in cross_vallidation_models.items():
    cv_scores = cross_val_score(model_object, x_cv_data, y_train, cv=5)
    mean_cv = cv_scores.mean()
    std_cv = cv_scores.std()

    cross_validation_summary[model_name] = (mean_cv, std_cv)

    print(f"{model_name:<30} mean={mean_cv:.4f} | std={std_cv:.4f}")
    interpret_cross_validation(model_name, mean_cv, std_cv)

most_accuracte_cv_model = max(cross_validation_summary, key=lambda name: cross_validation_summary[name][0])
most_stable_cv_model = min(cross_validation_summary, key=lambda name: cross_validation_summary[name][1])

print(f"\nMost accurate by CV mean: {most_accuracte_cv_model}")
print(f"Most stable by CV std: {most_accuracte_cv_model}")

print("\nOverall Interpretation:")
print("- The most accurate CV model had the highest average score across folds.")
print("- The most stable model had the lowest variability.")
print("- In practice, slightly less accurate but more stable model may sometimes be preferred.")
print("- This is especially true when you want predictable performance on new data.")

#================

#Task 5: Build Pipelines

#=================

print_section("Task 5: Build Predication Pipelines")

best_tree_model_name = "Random Forest" if random_forest_accuracy >= decision_tree_accuracy else "Decision Tree"
print(f"Best tree-based classifier: {best_tree_model_name}")

if best_tree_model_name =="Random Forest":
    best_tree_pipeline = Pipeline([
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
else:
    best_tree_pipeline = Pipeline([
        ("classifier", DecisionTreeClassifier(max_depth=chosen_tree_depth, random_state=42))
    ])

best_tree_pipeline.fit(x_train, y_train)
best_tree_pipeline_predictions = best_tree_pipeline.predict(x_test)

print("\n Best tree-based pipeline classification report:")
print(classification_report(y_test, best_tree_pipeline_predictions))

print("\nPipeline Interpretation:")
print("- Tree based models usually do not need scaling or PCA.")
print("- That is why this pipeline is simpler.")
print("- A simpler pipeline is easier to maintain when preprocessing is unnecessary.")

best_non_tree_scores = {
    "KNN scaled": knn_scaled_accuracy,
    "KNN PCA": knn_pca_accuracy,
    "Logistic Regression scaled": logistic_scaled_accuracy,
    "Logistic Regression PCA": logistic_pca_accuracy,
}

best_non_tree_model_name = max(best_non_tree_scores, key=best_non_tree_scores.get)
print(f"\nBest non-tree based classifier: {best_non_tree_model_name}")

if best_non_tree_model_name == "KNN scaled":
    best_non_tree_pipeline = Pipeline ([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=components_for_90, svd_solver="full")),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ])
elif best_non_tree_model_name == "KNN PCA":
    best_non_tree_pipeline = Pipeline ([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=components_for_90, svd_solver="full")),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ])

elif best_non_tree_model_name == "Logistic Regression scaled":
    best_non_tree_pipeline = Pipeline ([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))
    ])

else:
    best_non_tree_pipeline = Pipeline ([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=components_for_90, svd_solver="full"))
        ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))
    ])

best_non_tree_pipeline.fit(x_train, y_train)
best_non_tree_pipeline_predictions = best_non_tree_pipeline.predict(x_test)

print("\nBest non-tree based pipeline classification report:")
print(classification_report(y_test, best_non_tree_pipeline_predictions))

print("\nFinal Interpretation")
print("- Pipelines package preprocessing and the model into one reusable workflow.")
print("- This reduces mistakes because the same transformations used during training are applied during prediction.")
print("- Tree and non-tree pipelines do not need identical steps.")
print("- The correct pipeline depends on the needs of the model.")
print("- A good next improvement would be tuning hyperparameters instead of using default settings.")


