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


def _extract_first_available(
    dataset: xr.Dataset, candidates: list[str], shape: tuple[int, int]
) -> tuple[np.ndarray, str | None]:
    """Returns the first available 2D variable from candidates and its source name."""
    for name in candidates:
        if name in dataset.variables:
            arr = _extract_2d_or_nan(dataset, name, shape)
            return arr, name
    return np.full(shape, np.nan, dtype=float), None


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

    Args:
        filepath: Path to a local `.nc` file in ARGO format.

    Returns:
        A tuple `(cleaned_df, original_dataset)` where:
        - `cleaned_df` contains columns matching the DB schema subset:
          `float_id`, `date`, `latitude`, `longitude`, `depth_m`, `temp_c`, `salinity`.
        - `original_dataset` is the open xarray Dataset for downstream QC checks.

    Side effects:
        Opens the NetCDF file from disk.
    """
    dataset = xr.open_dataset(filepath)

    if "PRES" in dataset.variables:
        pres_2d = _normalize_numeric(_to_2d(np.asarray(dataset["PRES"].values)))
    elif "PRES_ADJUSTED" in dataset.variables:
        pres_2d = _normalize_numeric(_to_2d(np.asarray(dataset["PRES_ADJUSTED"].values)))
    else:
        raise ValueError("Could not find PRES or PRES_ADJUSTED in dataset.")
    n_prof, n_levels = pres_2d.shape

    temp_2d, _ = _extract_first_available(dataset, ["TEMP", "TEMP_ADJUSTED"], (n_prof, n_levels))
    sal_2d, sal_source = _extract_first_available(dataset, ["PSAL", "PSAL_ADJUSTED"], (n_prof, n_levels))
    temp_2d = _normalize_numeric(temp_2d)
    sal_2d = _normalize_numeric(sal_2d)

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
        }
    )

    # Always require temperature and depth; require salinity only if the source exists in file.
    drop_cols = ["temp_c", "depth_m", "date", "latitude", "longitude"]
    if sal_source is not None:
        drop_cols.append("salinity")

    cleaned_df = df.dropna(subset=drop_cols).reset_index(drop=True)
    return cleaned_df, dataset
