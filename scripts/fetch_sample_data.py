import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os


def fetch_sample_data() -> None:
    """Fetches a small ARGO sample region and saves per-float NetCDF files to data/raw/."""
    try:
        import argopy
        from argopy import DataFetcher as ArgoDataFetcher
    except ImportError:
        print("argopy is not installed. Please run: pip install argopy")
        return

    # Fetch a small box of real-time data from the Indian Ocean
    # box = [lon_min, lon_max, lat_min, lat_max, date_start, date_end]
    loader = ArgoDataFetcher(src="erddap").region(
        [60, 80, -20, 0, 0, 1000, "2023-01-01", "2023-03-01"]
    )

    # Download and save as NetCDF files to data/raw/
    ds = loader.to_xarray()

    # Split by float ID and save one file per float
    float_ids = ds["PLATFORM_NUMBER"].values
    unique_ids = list(
        set(
            [
                fid.decode().strip() if isinstance(fid, bytes) else str(fid).strip()
                for fid in float_ids
            ]
        )
    )

    os.makedirs("data/raw", exist_ok=True)

    for fid in unique_ids[:5]:  # limit to 5 floats for testing
        subset = ds.where(
            ds["PLATFORM_NUMBER"] == fid.encode()
            if isinstance(ds["PLATFORM_NUMBER"].values[0], bytes)
            else ds["PLATFORM_NUMBER"] == fid,
            drop=True,
        )
        out_path = f"data/raw/R{fid}_001.nc"
        subset.to_netcdf(out_path)
        print(f"Saved: {out_path}")

    print(f"Done. {len(unique_ids[:5])} files saved to data/raw/")


if __name__ == "__main__":
    fetch_sample_data()
