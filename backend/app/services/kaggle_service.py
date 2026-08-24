import os
import glob
import zipfile
import tempfile
import pandas as pd
from typing import Optional, Tuple
from app.schemas.profile import KaggleAuth

def set_kaggle_credentials(auth: Optional[KaggleAuth]):
    """Configures environment variables for Kaggle API authentication."""
    if auth and auth.username and auth.key:
        os.environ["KAGGLE_USERNAME"] = auth.username
        os.environ["KAGGLE_KEY"] = auth.key

def fetch_and_load_kaggle_dataset(
    dataset_slug: str,
    auth: Optional[KaggleAuth] = None,
    max_rows: int = 10000
) -> Tuple[pd.DataFrame, str]:
    """
    Downloads the primary CSV file from a Kaggle dataset using the official Kaggle API
    and loads it into a Pandas DataFrame.
    """
    set_kaggle_credentials(auth)
    
    # Import kaggle lazily after setting environment variables
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    temp_dir = tempfile.mkdtemp(prefix="dqaf_kaggle_")

    try:
        # Download and extract dataset files
        api.dataset_download_files(dataset_slug, path=temp_dir, unzip=True)

        # Locate tabular CSV files
        csv_files = glob.glob(os.path.join(temp_dir, "**", "*.csv"), recursive=True)
        if not csv_files:
            raise ValueError("No CSV files found in the specified Kaggle dataset.")

        # Pick the largest CSV file (usually the main dataset table)
        primary_csv = max(csv_files, key=os.path.getsize)
        file_name = os.path.basename(primary_csv)

        # Load into DataFrame (subsampling if large)
        df = pd.read_csv(primary_csv, nrows=max_rows)
        return df, file_name

    finally:
        # Clean up temporary files
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for file in files:
                try:
                    os.remove(os.path.join(root, file))
                except OSError:
                    pass
            for d in dirs:
                try:
                    os.rmdir(os.path.join(root, d))
                except OSError:
                    pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass