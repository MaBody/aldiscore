import os
from utils.path import WildcardPath
import pandas as pd


def get_dataset_dirs(parent_dir: WildcardPath, **wildcards: str):
    return list(
        filter(
            lambda name: not name.endswith("parquet"),
            os.listdir(parent_dir.format(**wildcards)),
        )
    )


def read_parquet_summary_file(parent_dir: WildcardPath, **wildcards: str):
    feat_file_path = parent_dir.format(**wildcards)
    df = pd.read_parquet(feat_file_path)
    df["path"] = feat_file_path
    df["benchmark"] = wildcards["benchmark"]
    return df


def format_summary_df(df):
    # Reformat to pivot the numeric features to individual columns
    feat_columns = df.select_dtypes(exclude="object").columns
    str_columns = df.select_dtypes(include="object").columns
    pivot_df = df.melt(id_vars=str_columns, value_vars=feat_columns)
    pivot_df["features"] = pivot_df["tool"] + "." + pivot_df["variable"]
    pivot_df["id"] = pivot_df["benchmark"] + "." + pivot_df["dataset"]
    pivot_df = pivot_df.pivot(
        index=["id"], columns="features", values="value"
    ).reset_index()
    return pivot_df


def load_balibase_results(
    dirs: list = ["balibase3/RV11", "balibase3/RV12"],
    benchmarking_msa=False,
    benchmarking_scores=False,
    with_reference=True,
):
    DATA_DIR = WildcardPath("/hits/fast/cme/bodynems/data/")
    OUTPUT_DIR = DATA_DIR / "output" / "{benchmark}"
    TOOL_DIR = OUTPUT_DIR / "{dataset}" / "{tool}"

    aligned_features_dfs = []
    unaligned_features_dfs = []
    reference_features_dfs = []
    confusion_features_dfs = []
    runtime_msa_dfs = []
    runtime_scores_dfs = []

    for benchmark in dirs:
        aligned_features = pd.read_parquet(
            OUTPUT_DIR.format(benchmark=benchmark) / "aligned_features.parquet"
        )
        unaligned_features = pd.read_parquet(
            OUTPUT_DIR.format(benchmark=benchmark) / "unaligned_features.parquet"
        )
        if with_reference:
            reference_features = pd.read_parquet(
                OUTPUT_DIR.format(benchmark=benchmark) / "reference_features.parquet"
            )
        confusion_features = pd.read_parquet(
            OUTPUT_DIR.format(benchmark=benchmark) / "confusion.parquet"
        )
        if benchmarking_msa:
            runtime_msa_df = pd.read_parquet(
                OUTPUT_DIR.format(benchmark=benchmark) / "runtime_msa.parquet"
            )
        if benchmarking_scores:
            runtime_scores_df = pd.read_parquet(
                OUTPUT_DIR.format(benchmark=benchmark) / "runtime_scores.parquet"
            )

        aligned_features["benchmark"] = benchmark
        aligned_features_dfs.append(aligned_features)

        unaligned_features["benchmark"] = benchmark
        unaligned_features_dfs.append(unaligned_features)

        if with_reference:
            reference_features["benchmark"] = benchmark
            reference_features_dfs.append(reference_features)

        confusion_features["benchmark"] = benchmark
        confusion_features_dfs.append(confusion_features)

        if benchmarking_msa:
            runtime_msa_df["benchmark"] = benchmark
            runtime_msa_dfs.append(runtime_msa_df)

        if benchmarking_scores:
            runtime_scores_df["benchmark"] = benchmark
            runtime_scores_dfs.append(runtime_scores_df)

    id_cols = ["benchmark", "dataset", "tool"]

    aligned_features = pd.concat(aligned_features_dfs, axis=0, ignore_index=True)
    aligned_cols = aligned_features.columns.copy()
    unaligned_features = pd.concat(unaligned_features_dfs, axis=0, ignore_index=True)
    if with_reference:
        reference_features = pd.concat(
            reference_features_dfs, axis=0, ignore_index=True
        )
    confusion_features = pd.concat(confusion_features_dfs, axis=0, ignore_index=True)

    merge_cols = id_cols[:-1] + ["data_type", "n_sequences", "max_sequence_length"]
    if benchmarking_msa:
        runtime_msa_df = pd.concat(runtime_msa_dfs, axis=0, ignore_index=True)

        runtime_msa_df = pd.merge(
            unaligned_features[merge_cols],
            runtime_msa_df,
            on=id_cols[:-1],
            how="outer",
        )
    if benchmarking_scores:
        runtime_scores_df = pd.concat(runtime_scores_dfs, axis=0, ignore_index=True)
        runtime_scores_df = pd.merge(
            unaligned_features[merge_cols],
            runtime_scores_df,
            on=id_cols[:-1],
            how="outer",
        )

    # aligned_features = aligned_features[aligned_features.benchmark == "balibase3/RV11"].reset_index(drop=True)
    # unaligned_features = unaligned_features[unaligned_features.benchmark == "balibase3/RV11"].reset_index(drop=True)
    # reference_features = reference_features[reference_features.benchmark == "balibase3/RV11"].reset_index(drop=True)
    # confusion_features = confusion_features[confusion_features.benchmark == "balibase3/RV11"].reset_index(drop=True)

    aligned_features = pd.merge(
        aligned_features,
        # unaligned_features[
        # [
        #     "dataset",
        #     "data_type",
        #     "n_sequences",
        #     "max_sequence_length",
        #     "lower_bound_gap_percentage",
        # ]
        # ],
        unaligned_features,
        how="outer",
        on=["benchmark", "dataset"],
    )
    if "mean_confusion_entropy" not in aligned_features.columns:
        aligned_features = pd.merge(
            aligned_features,
            confusion_features,
            how="inner",
            on=id_cols,
            validate="1:1",
        )

    if with_reference:
        print(reference_features.tool.value_counts())
        print(reference_features.dataset.value_counts().sort_values(ascending=True)[:5])
        reference_features = reference_features.groupby(id_cols, as_index=False).mean()
        reference_features = reference_features.sort_values(id_cols, ignore_index=True)

    aligned_features = aligned_features.sort_values(id_cols, ignore_index=True)
    confusion_features = confusion_features.sort_values(id_cols, ignore_index=True)

    print("unaligned:", unaligned_features.shape)
    print("aligned:", aligned_features.shape)
    print("confusion:", confusion_features.shape)
    if with_reference:
        print("reference:", reference_features.shape)

    return (
        unaligned_features,
        aligned_features,
        aligned_cols,
        reference_features if with_reference else None,
        confusion_features,
        runtime_msa_df if benchmarking_msa else None,
        runtime_scores_df if benchmarking_scores else None,
    )
