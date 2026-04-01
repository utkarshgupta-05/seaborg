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

    Side effects:
        None.
    """
    if df.empty:
        return df.copy()

    if "TEMP_QC" in dataset.variables:
        temp_qc_raw = np.asarray(dataset["TEMP_QC"].values)
        print(f"[QC DIAG] TEMP_QC dtype: {temp_qc_raw.dtype}")
        print(f"[QC DIAG] TEMP_QC first 10 raw values: {temp_qc_raw.reshape(-1)[:10]}")
        if temp_qc_raw.size > 0:
            print(f"[QC DIAG] TEMP_QC sample value type: {type(temp_qc_raw.reshape(-1)[0])}")
        else:
            print("[QC DIAG] TEMP_QC sample value type: <empty>")
    else:
        print("[QC DIAG] TEMP_QC missing in dataset.")

    if "PRES" in dataset.variables:
        pres_2d = _normalize_numeric(_to_2d(np.asarray(dataset["PRES"].values)))
    elif "PRES_ADJUSTED" in dataset.variables:
        pres_2d = _normalize_numeric(_to_2d(np.asarray(dataset["PRES_ADJUSTED"].values)))
    else:
        # If depth source is missing in dataset, do a safe range-only fallback on provided df.
        fallback_mask = (
            (df["temp_c"] >= -3.0)
            & (df["temp_c"] <= 40.0)
            & (df["salinity"] >= 20.0)
            & (df["salinity"] <= 42.0)
            & (df["depth_m"] > 0.0)
        )
        return df.loc[fallback_mask].reset_index(drop=True)

    n_prof, n_levels = pres_2d.shape
    n_rows_raw = n_prof * n_levels

    temp_2d = _extract_2d_numeric(dataset, ["TEMP", "TEMP_ADJUSTED"], (n_prof, n_levels))
    sal_2d = _extract_2d_numeric(dataset, ["PSAL", "PSAL_ADJUSTED"], (n_prof, n_levels))
    juld = pd.to_datetime(np.asarray(dataset["JULD"].values).reshape(-1), errors="coerce")
    lat = np.asarray(dataset["LATITUDE"].values).reshape(-1)
    lon = np.asarray(dataset["LONGITUDE"].values).reshape(-1)

    temp_qc = _flatten_qc(dataset, "TEMP_QC", n_rows_raw)
    psal_qc = _flatten_qc(dataset, "PSAL_QC", n_rows_raw)
    qc_good_mask_raw = (temp_qc == "1") & (psal_qc == "1")

    # Mirror parser dropna logic so QC flags align with parser output rows.
    temp_flat = temp_2d.reshape(-1)
    sal_flat = sal_2d.reshape(-1)
    depth_flat = pres_2d.reshape(-1)
    date_flat = np.repeat(juld.values, n_levels) if len(juld) == n_prof else np.full(n_rows_raw, np.datetime64("NaT"))
    lat_flat = np.repeat(lat, n_levels) if len(lat) == n_prof else np.full(n_rows_raw, np.nan)
    lon_flat = np.repeat(lon, n_levels) if len(lon) == n_prof else np.full(n_rows_raw, np.nan)

    parser_keep_raw = (
        ~pd.isna(temp_flat)
        & ~pd.isna(depth_flat)
        & ~pd.isna(date_flat)
        & ~pd.isna(lat_flat)
        & ~pd.isna(lon_flat)
    )
    if ("PSAL" in dataset.variables) or ("PSAL_ADJUSTED" in dataset.variables):
        parser_keep_raw = parser_keep_raw & ~pd.isna(sal_flat)

    aligned_qc_mask = qc_good_mask_raw[parser_keep_raw]

    # Safety alignment: never crash on shape drift; trim/pad to DataFrame length.
    if len(aligned_qc_mask) > len(df):
        aligned_qc_mask = aligned_qc_mask[: len(df)]
    elif len(aligned_qc_mask) < len(df):
        aligned_qc_mask = np.pad(
            aligned_qc_mask,
            (0, len(df) - len(aligned_qc_mask)),
            mode="constant",
            constant_values=True,
        )

    qc_df = df.loc[aligned_qc_mask].copy()

    in_range_mask = (
        (qc_df["temp_c"] >= -3.0)
        & (qc_df["temp_c"] <= 40.0)
        & (qc_df["salinity"] >= 20.0)
        & (qc_df["salinity"] <= 42.0)
        & (qc_df["depth_m"] > 0.0)
    )

    return qc_df.loc[in_range_mask].reset_index(drop=True)
