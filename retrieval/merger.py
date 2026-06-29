import pandas as pd

def merge_results(structured_df: pd.DataFrame, semantic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges structured and semantic retrieval rows, deduplicating them.
    Structured rows are given priority since they are authoritative.
    """
    combined_df = pd.concat([structured_df, semantic_df], ignore_index=True)
    if not combined_df.empty:
        # Ensure we have the subset columns before deduplicating
        subset_cols = [c for c in ["float_id", "date", "depth_m"] if c in combined_df.columns]
        if subset_cols:
            combined_df = combined_df.drop_duplicates(subset=subset_cols, keep="first")
    return combined_df
