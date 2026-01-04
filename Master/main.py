from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def resolve_base_path(cli_base: str | None) -> Path:
    """
    Prefer CLI base path. Otherwise use the directory containing this file.
    If that doesn't exist (not typical), fall back to current working directory.
    """
    if cli_base:
        return Path(cli_base).expanduser().resolve()

    try:
        return Path(__file__).resolve().parent
    except NameError:
        # __file__ may not exist in some interactive contexts
        return Path.cwd().resolve()


def load_hf_token(base_path: Path) -> str:
    """
    Loads token from .env. Supports both HUGGINGFACE_TOKEN and HF_TOKEN.
    """
    env_path = base_path / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))
        logger.info("Loaded environment from %s", env_path)
    else:
        logger.warning(".env not found at %s (will rely on OS env vars)", env_path)

    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Missing Hugging Face token. Set HUGGINGFACE_TOKEN (preferred) or HF_TOKEN.")
    return token


def run_job(job: str, base_path: Path, hf_token: str) -> int:
    """
    Dispatches pipeline jobs. Returns Unix-style exit code (0 success, 1 failure).
    """
    # Make sure the project root is importable
    sys.path.insert(0, str(base_path))

    if job == "register":
        from DataRegistration import DataRegistration
        obj = DataRegistration(str(base_path), hf_token)

    elif job == "prepare":
        from DataPrepration import DataPrepration
        obj = DataPrepration(str(base_path), hf_token)

    elif job == "modelbuilding":
        from BuildingModels import BuildingModels
        obj = BuildingModels(str(base_path), hf_token)

    elif job == "deploy":
        from HostingInHuggingFace import HostingInHuggingFace
        obj = HostingInHuggingFace(str(base_path), hf_token)

    else:
        logger.error("Unknown job: %s", job)
        return 1

    # Prefer new API (.run). Fall back to old (.ToRunPipeline).
    if hasattr(obj, "run"):
        ok = obj.run()
    elif hasattr(obj, "ToRunPipeline"):
        ok = obj.ToRunPipeline()
    else:
        logger.error("Selected class has neither run() nor ToRunPipeline().")
        return 1

    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a specific job in the pipeline")
    parser.add_argument(
        "--job",
        required=True,
        choices=["register", "prepare", "modelbuilding", "deploy"],
        help="Job to execute",
    )
    parser.add_argument(
        "--base-path",
        required=False,
        default=None,
        help="Project root path (defaults to directory containing main.py)",
    )
    args = parser.parse_args()

    base_path = resolve_base_path(args.base_path)
    logger.info("Base path: %s", base_path)

    # Ensure key folders exist (common expectation across pipeline)
    (base_path / "Data").mkdir(parents=True, exist_ok=True)
    (base_path / "Model_Dump_JOBLIB").mkdir(parents=True, exist_ok=True)
    (base_path / "Deployment").mkdir(parents=True, exist_ok=True)
    (base_path / "mlruns").mkdir(parents=True, exist_ok=True)

    hf_token = load_hf_token(base_path)
    exit_code = run_job(args.job, base_path, hf_token)

    if exit_code == 0:
        logger.info("Job '%s' completed successfully.", args.job)
    else:
        logger.error("Job '%s' failed.", args.job)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
