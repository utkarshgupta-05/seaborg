from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _to_2d(values: np.ndarray) -> np.ndarray:
    """Coerces profile-level or level arrays to shape (n_prof, n_levels)."""
    if values.ndim == 1:
        return values[:, np.newaxis]
    return values


def _extract_2d_or_nan(dataset: xr.Dataset, var_name: str, shape: tuple[int, int]) -> np.ndarray:
    """Returns a 2D numpy array for var_name or NaN array if missing."""
    if var_name not in dataset.variables:
        return np.full(shape, np.nan, dtype=float)

    values = np.asarray(dataset[var_name].values)
    values_2d = _to_2d(values)

    # If dimensions are transposed in an unusual file, try to coerce to expected shape.
    if values_2d.shape != shape and values_2d.T.shape == shape:
        values_2d = values_2d.T

    if values_2d.shape != shape:
        raise ValueError(f"Variable {var_name} has unexpected shape {values_2d.shape}, expected {shape}.")

    return values_2d


def _extract_merged_array_with_qc(
    dataset: xr.Dataset, base_name: str, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Extracts and merges adjusted and raw variables element-wise.
    Extracts corresponding QC flags exactly matching the value source.
    """
    has_any = False
    adj_val = np.full(shape, np.nan, dtype=float)
    raw_val = np.full(shape, np.nan, dtype=float)
    adj_qc = np.full(shape, b"1", dtype=object)  # default good
    raw_qc = np.full(shape, b"1", dtype=object)

    adj_name = f"{base_name}_ADJUSTED"
    adj_qc_name = f"{base_name}_ADJUSTED_QC"
    raw_qc_name = f"{base_name}_QC"

    if adj_name in dataset.variables:
        adj_val = _extract_2d_or_nan(dataset, adj_name, shape)
        has_any = True
        if adj_qc_name in dataset.variables:
            adj_qc = np.asarray(dataset[adj_qc_name].values)
            if adj_qc.ndim == 1:
                adj_qc = adj_qc[:, np.newaxis]
            if adj_qc.shape != shape and adj_qc.T.shape == shape:
                adj_qc = adj_qc.T

    if base_name in dataset.variables:
        raw_val = _extract_2d_or_nan(dataset, base_name, shape)
        has_any = True
        if raw_qc_name in dataset.variables:
            raw_qc = np.asarray(dataset[raw_qc_name].values)
            if raw_qc.ndim == 1:
                raw_qc = raw_qc[:, np.newaxis]
            if raw_qc.shape != shape and raw_qc.T.shape == shape:
                raw_qc = raw_qc.T

    # Element-wise merge: use raw if adj is NaN, otherwise use adj
    mask_use_raw = np.isnan(adj_val)
    merged_val = np.where(mask_use_raw, raw_val, adj_val)
    
    # Broadcast QC to match shape exactly before where()
    if adj_qc.shape != shape:
        adj_qc = np.full(shape, b"1", dtype=object)
    if raw_qc.shape != shape:
        raw_qc = np.full(shape, b"1", dtype=object)
        
    merged_qc = np.where(mask_use_raw, raw_qc, adj_qc)

    # Normalize QC strings
    def norm_qc(x):
        s = str(x).strip()
        if s.startswith("b'"):
            s = s[2:-1]
        s = s.replace("'", "")
        return s if s else "1"
    
    merged_qc_flat = np.array([norm_qc(x) for x in merged_qc.reshape(-1)], dtype=object)

    return merged_val, merged_qc_flat, has_any


def _normalize_numeric(values_2d: np.ndarray) -> np.ndarray:
    """
    Normalizes NetCDF numeric payloads by coercing masked/fill values to NaN.

    Many ARGO files use very large sentinel values for missing data.
    """
    arr = np.asarray(values_2d, dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    arr[np.abs(arr) > 1e20] = np.nan
    return arr


def parse_netcdf(filepath: str) -> tuple[pd.DataFrame, xr.Dataset]:
    """
    Parses one ARGO NetCDF file into a cleaned tabular DataFrame and the original dataset.
    """
    dataset = xr.open_dataset(filepath)

    if "N_PROF" not in dataset.dims or "N_LEVELS" not in dataset.dims:
        raise ValueError(f"Incompatible dimensions in {filepath}: Missing N_PROF or N_LEVELS.")

    n_prof = dataset.dims["N_PROF"]
    n_levels = dataset.dims["N_LEVELS"]

    if "PRES" not in dataset.variables and "PRES_ADJUSTED" not in dataset.variables:
        raise ValueError("Could not find PRES or PRES_ADJUSTED in dataset.")

    pres_2d, pres_qc, _ = _extract_merged_array_with_qc(dataset, "PRES", (n_prof, n_levels))
    pres_2d = _normalize_numeric(pres_2d)

    temp_2d, temp_qc, _ = _extract_merged_array_with_qc(dataset, "TEMP", (n_prof, n_levels))
    sal_2d, sal_qc, has_sal = _extract_merged_array_with_qc(dataset, "PSAL", (n_prof, n_levels))
    doxy_2d, doxy_qc, _ = _extract_merged_array_with_qc(dataset, "DOXY", (n_prof, n_levels))
    chla_2d, chla_qc, _ = _extract_merged_array_with_qc(dataset, "CHLA", (n_prof, n_levels))
    nitrate_2d, nitrate_qc, _ = _extract_merged_array_with_qc(dataset, "NITRATE", (n_prof, n_levels))

    temp_2d = _normalize_numeric(temp_2d)
    sal_2d = _normalize_numeric(sal_2d)
    doxy_2d = _normalize_numeric(doxy_2d)
    chla_2d = _normalize_numeric(chla_2d)
    nitrate_2d = _normalize_numeric(nitrate_2d)

    latitude = np.asarray(dataset["LATITUDE"].values).reshape(-1)
    longitude = np.asarray(dataset["LONGITUDE"].values).reshape(-1)
    juld = pd.to_datetime(np.asarray(dataset["JULD"].values).reshape(-1), errors="coerce")

    if len(latitude) != n_prof or len(longitude) != n_prof or len(juld) != n_prof:
        raise ValueError("Profile-level coordinate lengths do not match profile count.")

    file_name = Path(filepath).name
    float_id = file_name.split("_", maxsplit=1)[0]

    df = pd.DataFrame(
        {
            "float_id": np.repeat(float_id, n_prof * n_levels),
            "date": np.repeat(juld.values, n_levels),
            "latitude": np.repeat(latitude, n_levels),
            "longitude": np.repeat(longitude, n_levels),
            "depth_m": pres_2d.reshape(-1),
            "temp_c": temp_2d.reshape(-1),
            "salinity": sal_2d.reshape(-1),
            "oxygen": doxy_2d.reshape(-1),
            "chlorophyll": chla_2d.reshape(-1),
            "nitrate": nitrate_2d.reshape(-1),
            "_temp_qc": temp_qc,
            "_psal_qc": sal_qc,
            "_doxy_qc": doxy_qc,
            "_chla_qc": chla_qc,
            "_nitrate_qc": nitrate_qc,
        }
    )

    # Always require temperature and depth; require salinity only if the source exists in file.
    drop_cols = ["temp_c", "depth_m", "date", "latitude", "longitude"]
    if has_sal:
        drop_cols.append("salinity")

    cleaned_df = df.dropna(subset=drop_cols).reset_index(drop=True)
    return cleaned_df, dataset
