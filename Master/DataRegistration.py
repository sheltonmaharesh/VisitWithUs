
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class DataRegistrationConfig:
    base_path: str
    repo_id: str = "sheltonmaharesh/Tourism-visit-with-us-dataset"
    local_filename: str = "tourism.csv"
    repo_path: str = "Master/Data/tourism.csv"
    private: bool = False


class DataRegistration:
    """
    Creates or ensures a Hugging Face DATASET repository exists
    and uploads the local tourism.csv file into it.
    """

    def __init__(
        self,
        base_path: str,
        hf_token: Optional[str] = None,
        config: Optional[DataRegistrationConfig] = None,
    ) -> None:
        self.config = config or DataRegistrationConfig(base_path=base_path)
        self.hf_token = hf_token

        self.base_path = Path(self.config.base_path)
        self.data_dir = self.base_path / "Data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.api = HfApi(token=self.hf_token)

        logger.info("DataRegistration initialized")
        logger.info("Base path: %s", self.base_path)
        logger.info("Data dir: %s", self.data_dir)
        logger.info("HF dataset repo: %s", self.config.repo_id)

    def _source_file(self) -> Path:
        return self.data_dir / self.config.local_filename

    def create_repo_if_needed(self) -> bool:
        try:
            create_repo(
                repo_id=self.config.repo_id,
                repo_type="dataset",
                private=self.config.private,
                exist_ok=True,
                token=self.hf_token,
            )
            logger.info("Dataset repo ready: %s", self.config.repo_id)
            return True

        except HfHubHTTPError as e:
            logger.error("Hugging Face HTTP error: %s", e)
            return False
        except Exception as e:
            logger.exception("Unexpected error while creating repo: %s", e)
            return False

    def upload_source_data(self) -> bool:
        source = self._source_file()
        if not source.exists():
            logger.error("Source file not found: %s", source)
            return False

        try:
            self.api.upload_file(
                path_or_fileobj=str(source),
                path_in_repo=self.config.repo_path,
                repo_id=self.config.repo_id,
                repo_type="dataset",
                token=self.hf_token,
            )
            logger.info(
                "Uploaded %s to %s (%s)",
                source.name,
                self.config.repo_path,
                self.config.repo_id,
            )
            return True

        except HfHubHTTPError as e:
            logger.error("Hugging Face HTTP error during upload: %s", e)
            return False
        except Exception as e:
            logger.exception("Unexpected error during upload: %s", e)
            return False

    def run(self) -> bool:
        if not self.create_repo_if_needed():
            logger.error("Pipeline failed during repo creation")
            return False

        if not self.upload_source_data():
            logger.error("Pipeline failed during data upload")
            return False

        logger.info("Data registration pipeline completed successfully")
        return True
