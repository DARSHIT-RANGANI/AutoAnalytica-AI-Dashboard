import pandas as pd

def generate_dashboard_data(df: pd.DataFrame) -> dict:

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    dashboard = {}

    # ------------------------
    # 1. Summary Statistics
    # ------------------------
    dashboard["summary_stats"] = (
        df[numeric_cols].describe().to_dict()
        if numeric_cols else {}
    )

    # ------------------------
    # 2. Correlation Matrix
    # ------------------------
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        dashboard["correlation"] = corr.to_dict()
    else:
        dashboard["correlation"] = {}

    # ------------------------
    # 3. Top Categories (Bar Chart Data)
    # ------------------------
    category_data = {}

    for col in categorical_cols[:3]:  # limit to first 3
        counts = df[col].value_counts().head(10)
        category_data[col] = {
            "labels": counts.index.tolist(),
            "values": counts.values.tolist()
        }

    dashboard["categorical_distribution"] = category_data

    return dashboard