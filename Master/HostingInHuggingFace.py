
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class SpaceConfig:
    base_path: str
    space_repo_id: str = "sheltonmaharesh/Tourism-Prediction-Model-Space"
    deployment_dir: str = "Deployment"
    repo_type: str = "space"
    space_sdk: str = "docker"
    private: bool = False


class HostingInHuggingFace:
    """
    Creates/ensures a Hugging Face Space exists and uploads the local Deployment/ folder.
    """

    def __init__(self, base_path: str, hf_token: Optional[str] = None,
                 config: Optional[SpaceConfig] = None) -> None:
        self.config = config or SpaceConfig(base_path=base_path)
        self.hf_token = hf_token

        self.base_path = Path(self.config.base_path)
        self.deployment_path = self.base_path / self.config.deployment_dir

        self.api = HfApi(token=self.hf_token)

        logger.info("Initialized HostingInHuggingFace")
        logger.info("Space repo: %s", self.config.space_repo_id)
        logger.info("Deployment folder: %s", self.deployment_path)

    def create_space_if_needed(self) -> bool:
        """
        Ensures the Space repo exists.
        """
        try:
            self.api.repo_info(
                repo_id=self.config.space_repo_id,
                repo_type=self.config.repo_type,
                token=self.hf_token,
            )
            logger.info("Space already exists: %s", self.config.space_repo_id)
            return True

        except RepositoryNotFoundError:
            try:
                create_repo(
                    repo_id=self.config.space_repo_id,
                    repo_type=self.config.repo_type,
                    space_sdk=self.config.space_sdk,
                    private=self.config.private,
                    token=self.hf_token,
                    exist_ok=True,
                )
                logger.info("Space created: %s", self.config.space_repo_id)
                return True
            except Exception as e:
                logger.exception("Failed to create Space: %s", e)
                return False

        except HfHubHTTPError as e:
            logger.error("HF HTTP error while checking space: %s", e)
            return False
        except Exception as e:
            logger.exception("Unexpected error while checking space: %s", e)
            return False

    def upload_deployment_folder(self) -> bool:
        """
        Uploads the local Deployment/ directory into the Space repo.
        """
        if not self.deployment_path.exists() or not self.deployment_path.is_dir():
            logger.error("Deployment folder not found: %s", self.deployment_path)
            return False

        try:
            logger.info("Uploading folder %s -> %s", self.deployment_path, self.config.space_repo_id)
            self.api.upload_folder(
                repo_id=self.config.space_repo_id,
                folder_path=str(self.deployment_path),
                repo_type=self.config.repo_type,
                token=self.hf_token,
            )
            logger.info("Upload complete.")
            return True

        except HfHubHTTPError as e:
            logger.error("HF HTTP error while uploading deployment folder: %s", e)
            return False
        except Exception as e:
            logger.exception("Unexpected error while uploading deployment folder: %s", e)
            return False

    def run(self) -> bool:
        """
        Full pipeline:
        - Create Space if needed
        - Upload Deployment/ folder
        """
        if not self.create_space_if_needed():
            logger.error("Pipeline failed at Space creation/check.")
            return False

        if not self.upload_deployment_folder():
            logger.error("Pipeline failed at deployment upload.")
            return False

        logger.info("Deployment pipeline completed successfully.")
        return True
