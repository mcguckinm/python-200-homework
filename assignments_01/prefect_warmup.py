import numpy as np
import pandas as pd
from prefect import flow, task


@task
def create_series(arr: np.ndarray) -> pd.Series:
    """Convert numpy array to named pandas series"""
    return pd.Series(arr, name="values")

@task
def clean_data(series: pd.Series) -> pd.Series:
    """Drop missing and NAN values"""
    return series.dropna()

@task
def summaraize_data(series: pd.Series) -> dict:
    """summary stats for cleaned Series"""
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0],
    }

@flow
def pipeline_flow() -> dict:
    """Run the warmup pipeline using prefect tasks"""
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])
    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summaraize_data(cleaned)
    return summary
if __name__ == "__main__":
    result = pipeline_flow()
    print(result)


#question 1

#This workflow is easy and runs quickly because it is so short and uses it in a simple memory array
# for something this simple using plain python functions are easier to read, debug, and require fewer dependecies
# as well as less of an overall setup

#Question 2

#Prefect becomes more valuable when the same pipeline needs scheduling, retries, logging, 
# and more steps as well as specific runs, reliable execution is important. Simple logic becomes importnant especially 
# for something to run every day automatically. Loading files from cloud storage, or recover failures is important. 