def get_low_information_columns(df, dominance_threshold=0.95):
    low_info_cols = []
    for col in df.columns:
        top_freq = df[col].value_counts(normalize=True, dropna=False).values[0]
        if top_freq >= dominance_threshold:
            low_info_cols.append(col)
    return low_info_cols

def get_columns_with_excessive_nans(df, threshold=0.5):
    total_rows = len(df)
    drop_candidates = []
    for col in df.columns:
        nan_ratio = df[col].isna().sum() / total_rows
        if nan_ratio > threshold:
            drop_candidates.append(col)
    return drop_candidates