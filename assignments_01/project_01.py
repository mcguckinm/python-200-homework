"""World Happiness Prefect pipeline
it will save all outputs to assignments_01/outputs/"""

from __future__ import annotations
from typing import Any
from pathlib import Path
from venv import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from prefect import flow, get_run_logger, task
from scipy import stats
from scipy.stats import pearsonr

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_DIR = BASE_DIR.parent / "assignments" / "resources" / "happiness_project"

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names across yearly files."""
    renamed = {}

    for col in df.columns:
        clean = str(col).strip()
        lower = clean.lower()

        normalized = (
            lower.replace(".", " ")
                 .replace("-", " ")
                 .replace(":", " ")
                 .replace("_", " ")
                 .replace("(", "")
                 .replace(")", "")
                 .replace("+", " ")
        )
        normalized = " ".join(normalized.split())

        if normalized in {"country", "country name", "country or region"}:
            renamed[col] = "country"

        elif normalized in {"region", "regional indicator"}:
            renamed[col] = "region"

        elif normalized in {
            "happiness score",
            "score",
            "ladder score",
            "life ladder",
        }:
            renamed[col] = "happiness_score"

        elif normalized in {
            "overall rank",
            "rank",
        }:
            renamed[col] = "rank"

        elif normalized in {
            "economy gdp per capita",
            "logged gdp per capita",
            "gdp per capita",
            "explained by log gdp per capita",
        }:
            renamed[col] = "gdp_per_capita"

        elif normalized in {
            "family",
            "social support",
            "explained by social support",
        }:
            renamed[col] = "social_support"

        elif normalized in {
            "health life expectancy",
            "healthy life expectancy",
            "healthy life expectancy at birth",
            "explained by healthy life expectancy",
        }:
            renamed[col] = "healthy_life_expectancy"

        elif normalized in {
            "freedom",
            "freedom to make life choices",
            "explained by freedom to make life choices",
        }:
            renamed[col] = "freedom"

        elif normalized in {
            "generosity",
            "explained by generosity",
        }:
            renamed[col] = "generosity"

        elif normalized in {
            "trust government corruption",
            "perceptions of corruption",
            "explained by perceptions of corruption",
        }:
            renamed[col] = "corruption_perceptions"

        elif normalized in {
            "dystopia residual",
            "dystopia residual",
            "explained by dystopia residual",
        }:
            renamed[col] = "dystopia_residual"

        else:
            renamed[col] = clean

    df = df.rename(columns=renamed)

    # extra fallback aliases after rename
    fallback_map = {
        "Happiness Score": "happiness_score",
        "Score": "happiness_score",
        "Ladder score": "happiness_score",
        "Life Ladder": "happiness_score",
        "Country": "country",
        "Country name": "country",
        "Country or region": "country",
        "Regional indicator": "region",
        "Overall rank": "rank",
    }

    for old, new in fallback_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    return df

@task(retries=3, retry_delay_seconds=2)
def load_and_merge_data(data_dir:Path, output_dir: Path) -> pd.DataFrame:
    """load all happiness CSV"""

    logger = get_run_logger()

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files were found in {data_dir} Check dataset path in project_01.py"
        )

    frames: list[pd.DataFrame] = []
    for file_path in csv_files:
        year_text = "".join(ch for ch in file_path.stem if ch.isdigit())
        if not year_text:
            logger.info(f"Skipping file with no year in filename: {file_path.name}")
            continue

        year = int(year_text)
        logger.info(f"Loading {file_path.name} for year {year}")

        try:
            # First try normal comma-separated parsing
            df = pd.read_csv(file_path, sep=",", decimal=".")

            # Some files do not error, but load as one giant semicolon-packed column
            if len(df.columns) == 1 and ";" in str(df.columns[0]):
                logger.info(f"Re-reading {file_path.name} with semicolon separator")
                df = pd.read_csv(file_path, sep=";", decimal=",", engine="python")

        except pd.errors.ParserError:
            # 2017 is hitting this path
            logger.info(f"ParserError on {file_path.name}; retrying with semicolon separator")
            df = pd.read_csv(file_path, sep=";", decimal=",", engine="python")

        df = _normalize_columns(df)
        logger.info(f"{file_path.name} columns after normalize: {list(df.columns)}")

        df["year"] = year
        frames.append(df)

    if not frames:
        raise ValueError("No yearly data frames were loaded")

    merged_df = pd.concat(frames, ignore_index=True, sort=False)
    merged_df = _coerce_numeric_columns(merged_df)

    merged_path = output_dir / "merged_happiness.csv"
    merged_df.to_csv(merged_path, index=False)
    logger.info(f"Merged dataset saved to {merged_path}")
    logger.info(f"Merged data shape: {merged_df.shape}")
    logger.info(
    f"Non-null happiness_score counts by year: "
    f"{merged_df.groupby('year')['happiness_score'].count().to_dict()}"
)

    return merged_df

def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert likely numeric columns to numeric dtype."""
    df = df.copy()

    possible_numeric_cols = [
        "rank",
        "year",
        "happiness_score",
        "gdp_per_capita",
        "social_support",
        "healthy_life_expectancy",
        "freedom",
        "generosity",
        "corruption_perceptions",
        "dystopia_residual",
    ]

    for col in possible_numeric_cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.replace(",", ".", regex=False)
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

@task
def descriptive_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute and log descriptive statistics for happiness data."""
    logger = get_run_logger()
    df = _coerce_numeric_columns(df)

    df = df.copy()
    df["happiness_score"] = pd.to_numeric(df["happiness_score"], errors="coerce")

    score_series = df["happiness_score"].dropna()

    overall = {
        "mean": float(score_series.mean()),
        "median": float(score_series.median()),
        "std": float(score_series.std()),
    }
    logger.info(
        f"Overall happiness_score stats -> mean={overall['mean']:.4f}, "
        f"median={overall['median']:.4f}, std={overall['std']:.4f}"
    )

    by_year = (
        df.groupby("year", dropna=False)["happiness_score"]
        .mean()
        .sort_index()
        .to_dict()
    )
    logger.info(f"Mean happiness score by year: {by_year}")

    region_col = "region" if "region" in df.columns else None
    by_region = {}
    if region_col:
        by_region = (
            df.groupby(region_col, dropna=False)["happiness_score"]
            .mean()
            .sort_values(ascending=False)
            .to_dict()
        )
        logger.info(f"Mean happiness score by region: {by_region}")
    else:
        logger.info("No normalized region column found, so region statistics were skipped.")

    return {"overall": overall, "by_year": by_year, "by_region": by_region}
@task
def create_visualizations(df:pd.DataFrame, output_dir: Path) -> dict[str, str]:
    """create and save all visualizations"""
    df = _coerce_numeric_columns(df)

    logger = get_run_logger()
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # histogram
    plt.figure(figsize=(10, 5))
    plt.hist(df["happiness_score"], bins=20, edgecolor="black")
    plt.title("Happiness Score by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    box_path = output_dir /"happiness_by_year.png"
    plt.tight_layout()
    plt.savefig(box_path)
    plt.close()
    saved_files["boxplot"] = str(box_path)
    logger.info(f"Saved yearly boxplot to {box_path}")

    #Scatter GDP vs happiness
    if "gdp_per_capita" in df.columns:
        plt.figure(figsize=(8,5))
        plt.scatter(df["gdp_per_capita"], df["happiness_score"], alpha=0.6)
        plt.title("GDP per Capita vs Happiness Score")
        plt.xlabel("GDP per Capita")
        plt.ylabel("Happiness Score")
        scatter_path = output_dir / "gdp_vs_happiness.png"
        plt.tight_layout()
        plt.savefig(scatter_path)
        plt.close()
        saved_files["scatter"] = str(scatter_path)
        logger.info(f"Saved GDP vs happiness scatter plot to {scatter_path}")
    else:
        logger.info(f"gdp_per_capita column not found it was skipped")

    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10,8))
    sns.heatmap(numeric_df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    heatmap_path = output_dir / "correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    saved_files["heatmap"] = str(heatmap_path)
    logger.info(f"Saved Correlation heatmap to {heatmap_path}")

    return saved_files


@task
def run_hypothesis_test(df: pd.DataFrame) -> dict[str, Any]:
    """Run required tests"""
    df = _coerce_numeric_columns(df)
    logger = get_run_logger()

    data_2019 = df.loc[df["year"] == 2019, "happiness_score"].dropna()
    data_2020 = df.loc[df["year"] == 2020, "happiness_score"].dropna()
    if data_2019.empty or data_2020.empty:
        raise ValueError("Could not find happiness_score data for 2019 and 2020.")

    t_stat, p_value = stats.ttest_ind(data_2019, data_2020, equal_var=False)
    mean_2019 = float(data_2019.mean())
    mean_2020 = float(data_2020.mean())

    if p_value < 0.05:
        interpretation = (
            f"The average happiness score changed from 2019 ({mean_2019:.3f})"
            f" to 2020({mean_2020:.3f}), but the difference is unlikely to be due to chance alone."
        )
    else:
        interpretation = (
            f"The average happiness score changed from 2019 ({mean_2019:.3f}) "
            f"to 2020 ({mean_2020:.3f}), but the difference is not statistically convincing at alpha = 0.05"
        )

    logger.info(f"2019 vs 2020 t-test -> t={t_stat:.4f}, p={p_value:.6f}")
    logger.info(f"2019 mean happiness: {mean_2019:.4f}")
    logger.info(f"2020 mean happiness: {mean_2020:.4f}")
    logger.info(interpretation)

    second_test: dict[str, Any] = {
        "description": "No second test was run",
        "t_stat": None,
        "p_value": None,
    }

    if "region" in df.columns:
        region_means = (
            df.groupby("region")["happiness_score"]
            .mean()
            .sort_values(ascending=False)

        )

        valid_regions = [idx for idx in region_means.index if pd.notna(idx)]
        if len(valid_regions) >= 2:
            top_region = valid_regions[0]
            bottom_region = valid_regions[-1]

            top_scores = df.loc[df["region"] == top_region, "happiness_score"].dropna()
            bottom_scores = df.loc[df["region"] == bottom_region, "happiness_score"].dropna()

            if not top_scores.empty and not bottom_scores.empty:
                t_stat2, p_value2 = stats.ttest_ind(top_scores, bottom_scores, equal_var=False)
                second_test = {
                    "description": f"T-test comparing happiness scores between the highest scoring region ({top_region}) and the lowest scoring region ({bottom_region}).",
                    "t_stat": t_stat2,
                    "p_value": p_value2,
                    "group1": top_region,
                    "group2": bottom_region,
                    "mean1": float(top_scores.mean()),
                    "mean2": float(bottom_scores.mean()),
                }
                logger.info(f"Second t-test ({top_region} vs {bottom_region}) -> t={t_stat2:.4f}, p={p_value2:.6f}")
    return {
        "pre_post_2020": {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "mean_2019": mean_2019,
            "mean_2020": mean_2020,
            "interpretation": interpretation,
            },
            "second_test": second_test,
            }


@task
def run_correlation_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Run required correlation analysis"""
    df = _coerce_numeric_columns(df)

    logger = get_run_logger()

    if "happiness_score" not in df.columns:
        raise KeyError("Merged data does not contain a happiness_score column")

    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != "happiness_score"]

    results = []

    for col in numeric_cols:
        temp = df[[col, "happiness_score"]].dropna()
        if len(temp) < 3:
            logger.info(f"Skipping correlation {col} because there are not enough non-null rows")
            continue
    r_value, p_value = pearsonr(temp[col], temp["happiness_score"])
    result = {
        "variable": col,
        "r_value": float(r_value),
        "p_value": float(p_value),
    }
    results.append(result)
    logger.info(f"Correlation between {col} and happiness_score -> r={r_value:.4f}, p={p_value:.6f}")


    number_of_tests = len(results)
    if number_of_tests == 0:
        raise ValueError(" No correlation tests could be computed")

    adjusted_alpha = 0.05 / number_of_tests
    logger.info(f"Bonferroni adjusted alpha for {number_of_tests} tests: {adjusted_alpha:.6f}")

    sig_original = []
    sig_bonferroni = []
    for result in results:
        result["significant_original"] = result["p_value"] < 0.05
        result["significant_bonferroni"] = result["p_value"] < adjusted_alpha

        if result["significant_original"]:
            sig_original.append(result["variable"])
        if result["significant_bonferroni"]:
            sig_bonferroni.append(result["variable"])

    logger.info(f"Significant at alpha=0.05: {sig_original}")
    logger.info(f"Significant at Bonferroni adjusted alpha: {sig_bonferroni}")

    bonf_results = [r for r in results if r["significant_bonferroni"]]

    if bonf_results:
        strongest = max(bonf_results, key=lambda x: abs(x["r_value"]))
    else:
        strongest = max(results, key=lambda x: abs(x["r"]))


    return {
        "results": results,
        "number_of_tests": number_of_tests,
        "adjusted_alpha": adjusted_alpha,
        "significant_original": sig_original,
        "significant_bonferroni": sig_bonferroni,
        "strongest_variable": strongest,
    }


@task
def summary_report(
    df: pd.DataFrame,
    descriptive_results: dict[str, Any],
    hypothesis_results: dict[str, Any],
    correlation_results: dict[str, Any],
) -> None:
    """ log human readable summary of pipeline"""

    logger = get_run_logger()

    country_col = "country" if "country" in df.columns else None
    num_countries = int(df[country_col].nunique() if country_col else int(len(df)))

    num_years = int(df["year"].nunique())

    logger.info(f"Merged dataset contains {num_countries} unique countries across {num_years} years.")


    by_region = descriptive_results.get("by_region", {} )
    if by_region:
        region_series = pd.Series(by_region).sort_values(ascending=False)
        top_3 = region_series.head(3).to_dict()
        bottom_3 = region_series.tail(3).to_dict()
        logger.info(f"Top 3 happiest regions: {top_3}")
        logger.info(f"Bottom 3 happiest regions: {bottom_3}")
    else:
        logger.info("No region data available to summarize top and bottom regions")

    logger.info(
        f"pre/post-2020 result: {hypothesis_results['pre_post_2020']['interpretation']}"
    )

    strongest = correlation_results["strongest_variable"]
    bonf_sig = strongest.get("significant_bonferroni", False)

    logger.info(
        "Most strongly correlated variable with happiness_score "
        f"{ 'after Bonferroni correction' if bonf_sig else '(strongest overall; no Bonferroni-significant result available)'}: "
        f"{strongest['variable']} (r={strongest['r_value']:.4f}, p={strongest['p_value']:.6f})"
    )

@flow
def happiness_pipeline():
    """main flow to run all tasks"""

    merged_df = load_and_merge_data(DATA_DIR, OUTPUT_DIR)
    descriptive_results = descriptive_statistics(merged_df)
    create_visualizations(merged_df, OUTPUT_DIR)
    hypothesis_results = run_hypothesis_test(merged_df)
    correlation_results = run_correlation_analysis(merged_df)
    summary_report(merged_df, descriptive_results, hypothesis_results, correlation_results)


if __name__ == "__main__":
    happiness_pipeline()