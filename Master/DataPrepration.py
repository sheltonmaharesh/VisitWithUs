
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class DataPreparationConfig:
    base_path: str
    repo_id: str = "sheltonmaharesh/Tourism-visit-with-us-dataset"
    source_repo_file: str = "Master/Data/tourism.csv"
    data_subdir: str = "Data"
    target_col: str = "ProdTaken"
    test_size: float = 0.2
    random_state: int = 42
    train_filename: str = "train.csv"
    test_filename: str = "test.csv"


class DataPrepration:
    """
    Loads tourism.csv from a Hugging Face dataset repo, cleans it, splits into train/test,
    saves locally, and uploads train/test back into the same repo.
    """

    def __init__(self, base_path: str, hf_token: Optional[str] = None,
                 config: Optional[DataPreparationConfig] = None) -> None:
        self.config = config or DataPreparationConfig(base_path=base_path)
        self.hf_token = hf_token

        self.base_path = Path(self.config.base_path)
        self.data_dir = self.base_path / self.config.data_subdir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.api = HfApi(token=self.hf_token)

        logger.info("Initialized DataPrepration")
        logger.info("Base path: %s", self.base_path)
        logger.info("Local data dir: %s", self.data_dir)
        logger.info("HF dataset repo: %s", self.config.repo_id)

    def load_dataset_from_hf(self) -> Optional[pd.DataFrame]:
        """
        Downloads the source CSV from HF and returns a DataFrame.
        """
        try:
            local_path = hf_hub_download(
                repo_id=self.config.repo_id,
                filename=self.config.source_repo_file,
                repo_type="dataset",
                token=self.hf_token,
            )
            df = pd.read_csv(local_path)

            # Drop junk index column if it exists
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])

            logger.info("Loaded dataset from HF: %s", self.config.source_repo_file)
            logger.info("Dataset shape: %s", df.shape)
            return df

        except FileNotFoundError:
            logger.error("Source file not found in repo: %s", self.config.source_repo_file)
            return None
        except HfHubHTTPError as e:
            logger.error("HF HTTP error while downloading: %s", e)
            return None
        except Exception as e:
            logger.exception("Unexpected error while loading dataset: %s", e)
            return None

    def train_test_split_df(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Splits into train/test with stratification if target exists and is valid.
        """
        try:
            if self.config.target_col not in df.columns:
                logger.error("Target column '%s' not found in dataset.", self.config.target_col)
                return None, None

            y = df[self.config.target_col]
            can_stratify = y.notna().all() and y.nunique() > 1

            if can_stratify:
                stratify = y
                logger.info("Using stratified split on '%s'.", self.config.target_col)
            else:
                stratify = None
                logger.warning("Not stratifying (target missing/constant/has NaNs).")

            df_train, df_test = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=stratify,
                shuffle=True,
            )

            logger.info("Train shape: %s | Test shape: %s", df_train.shape, df_test.shape)
            return df_train, df_test

        except Exception as e:
            logger.exception("Unexpected error during train/test split: %s", e)
            return None, None

    def dataset_cleaning(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Cleans the dataset:
        - Fix known dirty categorical values
        - Drop duplicate CustomerID rows (if present)
        - Fill missing numeric with median, categorical with mode
        - Drop CustomerID column (if present) to avoid leakage/ID usage
        """
        try:
            out = df.copy(deep=True)

            # Fix known typo if column exists
            if "Gender" in out.columns:
                out["Gender"] = out["Gender"].replace({"Fe Male": "Female"})

            # Drop duplicates by CustomerID if available
            if "CustomerID" in out.columns:
                out = out.drop_duplicates(subset=["CustomerID"], keep="first").reset_index(drop=True)

            # Fill missing values
            numeric_cols = out.select_dtypes(include=["number"]).columns
            cat_cols = [c for c in out.columns if c not in numeric_cols]

            for col in numeric_cols:
                if out[col].isna().any():
                    out[col] = out[col].fillna(out[col].median())

            for col in cat_cols:
                if out[col].isna().any():
                    mode_vals = out[col].mode(dropna=True)
                    fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
                    out[col] = out[col].fillna(fill_val)

            # Drop ID column (if present)
            if "CustomerID" in out.columns:
                out = out.drop(columns=["CustomerID"])

            return out

        except Exception as e:
            logger.exception("Unexpected error during cleaning: %s", e)
            return None

    def upload_into_hf(self, df: pd.DataFrame, filename: str) -> bool:
        """
        Saves df locally and uploads it to HF repo under Master/Data/{filename}.
        """
        try:
            local_file = self.data_dir / filename
            df.to_csv(local_file, index=False)

            self.api.upload_file(
                path_or_fileobj=str(local_file),
                path_in_repo=f"Master/Data/{filename}",
                repo_id=self.config.repo_id,
                repo_type="dataset",
                token=self.hf_token,
            )

            logger.info("Uploaded %s to HF repo: %s", filename, self.config.repo_id)
            return True

        except HfHubHTTPError as e:
            logger.error("HF HTTP error during upload of %s: %s", filename, e)
            return False
        except Exception as e:
            logger.exception("Unexpected error during upload of %s: %s", filename, e)
            return False

    def run(self) -> bool:
        """
        Full pipeline:
        - Load tourism.csv from HF
        - Split train/test
        - Clean both
        - Save locally + upload train.csv and test.csv back to HF
        """
        df = self.load_dataset_from_hf()
        if df is None:
            return False

        df_train, df_test = self.train_test_split_df(df)
        if df_train is None or df_test is None:
            return False

        df_train_clean = self.dataset_cleaning(df_train)
        df_test_clean = self.dataset_cleaning(df_test)
        if df_train_clean is None or df_test_clean is None:
            return False

        ok_train = self.upload_into_hf(df_train_clean, self.config.train_filename)
        ok_test = self.upload_into_hf(df_test_clean, self.config.test_filename)

        if not (ok_train and ok_test):
            logger.error("Pipeline failed during upload stage.")
            return False

        logger.info("Data preparation completed successfully.")
        return True
