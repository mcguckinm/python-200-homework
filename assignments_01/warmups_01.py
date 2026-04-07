import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns


#Pandas Question 1

data = {
    "name": ["Alice","Bob", "Carol", "David", "Eve"],
    "grade": [85, 72, 90, 68, 95],
    "city": ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}

df = pd.DataFrame(data)

print(f"Num Rows: {len(df)}")

print(f"First three rows: {df.head(3)}")
print(f"Shape: {df.shape}")
print(f"Type: {df.dtypes}")
print(f"\nQuestion 2")
filtered_df = df[(df["passed"]) & (df["grade"]> 80)]

print(filtered_df)

print(f"\nQuestion 3:")
df["grade_curved"] = df["grade"] + 5
print(df)

print(f"\nQuestion4:")

df["name_upper"] = df["name"].str.upper()
print(df[["name", "name_upper"]])

print(f"\nQuestion 5:")

city_mean = df.groupby("city")["grade"].mean()

print(city_mean)

print(f"\nQuestion 6:")

df["city"] = df["city"].replace("Austin", "Houston")
print(df[["name", "city"]])

print(f"\nQuestion 7:")

sorted_df = df.sort_values(by="grade", ascending=False)

print(sorted_df.head(3))


arr_1d = np.array([10, 20, 30, 40 ,50])

print(f"\nNumpy Question 1:")
print(f"Shape: {arr_1d.shape}")
print(f"Dtype: {arr_1d.dtype}")
print(f"Ndim: {arr_1d.ndim}")


print(f"\nNumpy Question 2")
arr = np.array([[1, 2, 3],
                [4,5,6],
                [7, 8, 9]])

print(f"Shape: {arr.shape}")
print(f"Size: {arr.size}")

print(f"\nNumpy Question 3")
print(arr[:2,:2])

print(f"\nNumpy Question 4")

zeros_arr = np.zeros((3,4))
ones_arr = np.ones((2,5))

print("Zeros array:")
print(zeros_arr)
print("Ones array:")
print(ones_arr)

print(f"\nQuestion 5:")

np_range = np.arange(0,50,5)

print(np_range)
print(f"Shape: {np_range.shape}")
print(f"Mean: {np_range.mean()}")
print(f"Sum: {np_range.sum()}")
print(f"Standard Deviation: {np_range.std()}")

print(f"\nQuestion 6:")

np_random = np.random.normal(0,1,200)

print(f"Mean: {np_random.mean()}")
print(f"Standard Deviation: {np_random.std()}")


print(f"Matplotlib review")

print(f"\nQuestion 1")

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.figure()
plt.plot(x,y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

print(f"\nQuestion 2")

subjects = ["Math", "Science", "English", "Hisotry"]
scores = [88, 92, 75, 83]
plt.figure()
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subject")
plt.ylabel("Score")
plt.show()


print(f"\nQuestion 3:")

x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.figure()
plt.scatter(x1, y1, label="Dataset 1")
plt.scatter(x2, y2, label="Dataset 2")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

print(f"\nQuestion 4:")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(x, y)
axes[0].set_title("Squares")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[1].bar(subjects, scores)
axes[1].set_title("Subject Scores")
axes[1].set_xlabel("Subject")
axes[1].set_ylabel("Score")
plt.tight_layout()
plt.show()


#Descriptive Statistics

print(f"Descriptive Statistics Review")


print(f"Question 1:")
data_stats = [12, 15, 14, 10, 18, 22, 13, 16, 14, 16]
print(f"Mean: {np.mean(data_stats)}")
print(f"Median: {np.median(data_stats)}")
print(f"Variance: {np.var(data_stats)}")
print(f"Standard Deviation: {np.std(data_stats)}")

print(f"\nQuestion 2")

random_scores = np.random.normal(65, 10, 500)
plt.figure()
plt.hist(random_scores, bins=50)
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.show()



print(f"\nQuestion 3")

group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.figure()
plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.show()

print(f"\nQuestion 4:")
normal_data = np.random.normal(50,5,200)
skewed_data = np.random.exponential(10, 200)
plt.figure()
plt.boxplot([normal_data, skewed_data], tick_labels=["Normal","Exponential"])
plt.title("Distribution Comparison")
plt.show()
#exponential data is more skewed. Normal data shows mean is in the middle which is usually appropriate measure to find the mean.
#Meanwhile exponential data media is often more appropriate because it is less affected.

print(f"Question 5:")
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]
mode1 = stats.mode(data1, keepdims=False)
mode2 = stats.mode(data2, keepdims=False)
print("Data 1:")
print(f"Mean: {np.mean(data1)}")
print(f"Median: {np.median(data1)}")
print(f"Mode: {mode1.mode}")
print(f"Data 2:")
print(f"Mean: {np.mean(data2)}")
print(f"Median: {np.median(data2)}")
print(f"Mode: {mode2.mode}")

#simple answer is when looking at the numbers for our Mode the last number we have listed for data1 is 18 where as data2 is 150


#Hypothesis Testing

print(f"\nHypothesis Testing")

print(f"\nQuestion 1")
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")

print(f"\nQuestion 2:")

if(p_value < 0.05):
    print("The result is statistically significant at alpha = 0.05.")
else:
    print("The result is not statistically significant at alpha = 0.05.")

print(f"\nQuestion 3")
before = [60, 65, 70, 58, 62, 67, 63, 66]
after = [68, 70, 76, 65, 69, 72, 70, 71]
paired_t, paired_p = stats.ttest_rel(before, after)
print(f"t-statistic: {paired_t}")
print(f"p-value: {paired_p}")

print(f"\nQuestion 4")
scores = [72, 68, 75, 70, 69, 74, 71, 73]

one_sample_t, one_sample_p = stats.ttest_1samp(scores, popmean=70)
print(f"t-statistics: {one_sample_t}")
print(f"p-value: {one_sample_p}")

print(f"\nQuestion 5")

one_tailed_t, one_tailed_p = stats.ttest_ind(group_a, group_b, alternative="less")
print(f"One-tailed t-statistic {one_tailed_t}")
print(f"One-tailed p-value {one_tailed_p}")

print(f"\nQuestion 6")
print(f"\nStudents in group b scored high on average than students in group a. Difference is unlikely based  on random chance.")

#correlation

print(f"\nCorrelation")

print(f"\nCorrelation Q1")
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x,y)

print(f"Full correlation  matrix:")
print(corr_matrix)

print(f"Correlation Coefficient [0, 1]: {corr_matrix[0, 1]}")

# I expect the correlation to increase in a steady line. Even imagining the line chart shows steady increase in the x and y axis. 
# data initally shows a lineaer growth and that is what is expected. 

print(f"\nCorrelation Q2")
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

r_value, p_value = pearsonr(x, y)
print(f"Correlation Coefficient: {r_value}")
print(f"p-value: {p_value}")

print("Correlation Q3")
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55, 60, 65, 72, 80],
    "age": [25, 30, 22, 35, 28],
 }

people_df = pd.DataFrame(people)

people_corr = people_df.corr()
print(people_corr)

print("Correlation Q4")
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

plt.figure()
plt.scatter(x,y)
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

print("Correlation Q5")

plt.figure()
sns.heatmap(people_corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()

#pipelines

print("\nPipelines")

print("Pipeline Q1")
pipeline_arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr: np.ndarray) -> pd.Series:
    """converting numpy array into pd array"""
    return pd.Series(arr, name="values")

def clean_data(series: pd.Series) -> pd.Series:
    """Dropping NAN values/ missing values from series"""
    return series.dropna()

def summarize_data(series: pd.Series) -> dict:
    """return summary stats for cleaned series"""
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode":series.mode()[0],
    }

def data_pipeline(arr: np.ndarray) -> dict:
    """Run the plain python data pipeline from start to finish"""
    series = create_series(pipeline_arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    return summary

result = data_pipeline(pipeline_arr)

for key, value in result.items():
    print(f"{key}: {value}")