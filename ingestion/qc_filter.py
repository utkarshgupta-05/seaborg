import numpy as np
import pandas as pd
import xarray as xr


def _to_2d(values: np.ndarray) -> np.ndarray:
    """Coerces 1D or 2D variable payloads to (n_prof, n_levels) layout."""
    if values.ndim == 1:
        return values[:, np.newaxis]
    return values


def _normalize_numeric(values_2d: np.ndarray) -> np.ndarray:
    """Converts fill/sentinel values to NaN for consistent parser-like filtering."""
    arr = np.asarray(values_2d, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    arr[np.abs(arr) > 1e20] = np.nan
    return arr


def _extract_2d_numeric(dataset: xr.Dataset, candidates: list[str], shape: tuple[int, int]) -> np.ndarray:
    """Returns first available numeric variable from candidates in expected shape."""
    for name in candidates:
        if name in dataset.variables:
            arr = _to_2d(np.asarray(dataset[name].values))
            if arr.shape != shape and arr.T.shape == shape:
                arr = arr.T
            if arr.shape == shape:
                return _normalize_numeric(arr)
    return np.full(shape, np.nan, dtype=float)


def _flatten_qc(dataset: xr.Dataset, var_name: str, expected_len: int) -> np.ndarray:
    """Flattens a QC variable to 1D string array; if missing, default to good ('1')."""
    if var_name not in dataset.variables:
        return np.full(expected_len, "1", dtype=object)

    raw = np.asarray(dataset[var_name].values).reshape(-1)
    normalized = []
    for item in raw:
        if isinstance(item, (bytes, bytearray)):
            normalized.append(item.decode("utf-8", errors="ignore").strip())
        else:
            normalized.append(str(item).strip())
    out = np.asarray(normalized, dtype=object)

    # Length safety: trim/pad so downstream masking never crashes.
    if len(out) > expected_len:
        out = out[:expected_len]
    elif len(out) < expected_len:
        out = np.pad(out, (0, expected_len - len(out)), mode="constant", constant_values="1")
    return out


def apply_qc(df: pd.DataFrame, dataset: xr.Dataset) -> pd.DataFrame:
    """
    Applies ARGO QC flags and scientific range checks to parsed ARGO rows.

    Args:
        df: Parsed DataFrame from `parser.parse_netcdf()`.
        dataset: Original xarray Dataset from `parser.parse_netcdf()`.

    Returns:
        Filtered DataFrame containing only scientifically valid rows.
    """
    if df.empty:
        return df.copy()

    VALID_QC_FLAGS = {"1", "2"}

    # Base mask for temperature and salinity row-level rejection
    temp_qc_good = df["_temp_qc"].isin(VALID_QC_FLAGS)
    qc_good_mask_raw = temp_qc_good.copy()
    
    # Only reject by salinity QC if the file actually contained salinity source
    has_salinity = ("PSAL" in dataset.variables) or ("PSAL_ADJUSTED" in dataset.variables)
    if has_salinity:
        psal_qc_good = df["_psal_qc"].isin(VALID_QC_FLAGS)
        qc_good_mask_raw = qc_good_mask_raw & psal_qc_good

    qc_df = df.loc[qc_good_mask_raw].copy()

    # Apply scientific range checks for row rejection
    in_range_mask = (
        (qc_df["temp_c"] >= -3.0)
        & (qc_df["temp_c"] <= 40.0)
        & (qc_df["depth_m"] > 0.0)
    )
    if has_salinity:
        in_range_mask = in_range_mask & (qc_df["salinity"] >= 20.0) & (qc_df["salinity"] <= 42.0)

    qc_df = qc_df.loc[in_range_mask].copy()

    # BGC variables: null out the value if QC is bad, do NOT drop the row
    BGC_QC_MAP = {
        "oxygen": "_doxy_qc",
        "chlorophyll": "_chla_qc",
        "nitrate": "_nitrate_qc",
    }
    
    for bgc_var, qc_col in BGC_QC_MAP.items():
        if qc_col in qc_df.columns:
            bad_bgc_mask = ~qc_df[qc_col].isin(VALID_QC_FLAGS)
            qc_df.loc[bad_bgc_mask, bgc_var] = np.nan

    # Drop temporary QC columns before returning
    cols_to_drop = [c for c in qc_df.columns if c.endswith("_qc")]
    qc_df = qc_df.drop(columns=cols_to_drop).reset_index(drop=True)

    return qc_df
