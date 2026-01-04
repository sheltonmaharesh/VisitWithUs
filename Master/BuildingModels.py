
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class ModelConfig:
    base_path: str
    hf_token: Optional[str] = None

    # HF repos (owner fixed)
    ds_repo_id: str = "sheltonmaharesh/Tourism-visit-with-us-dataset"
    model_repo_id: str = "sheltonmaharesh/Tourism_Prediction_Model"

    # HF file paths
    train_file: str = "Master/Data/train.csv"
    test_file: str = "Master/Data/test.csv"

    # local folders
    model_dir_name: str = "Model_Dump_JOBLIB"
    mlruns_dir_name: str = "mlruns"

    # experiment
    mlflow_experiment: str = "Tourism-Prediction-Experiment"

    # split/eval
    cv_splits: int = 3
    random_state: int = 42
    n_iter_search: int = 30


class BuildingModels:
    def __init__(self, base_path: str, hf_token: Optional[str] = None,
                 config: Optional[ModelConfig] = None) -> None:
        self.config = config or ModelConfig(base_path=base_path, hf_token=hf_token)
        self.base_path = Path(self.config.base_path)
        self.hf_token = self.config.hf_token

        self.api = HfApi(token=self.hf_token)

        # Local dirs
        self.model_dir = self.base_path / self.config.model_dir_name
        self.model_dir.mkdir(parents=True, exist_ok=True)

        mlruns_path = self.base_path / self.config.mlruns_dir_name
        mlruns_path.mkdir(parents=True, exist_ok=True)

        # MLflow local tracking
        mlflow.set_tracking_uri(f"file://{mlruns_path}")
        mlflow.set_experiment(self.config.mlflow_experiment)

        # Columns
        self.target_col = "ProdTaken"
        self.categorical_columns = [
            "TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation"
        ]
        self.numerical_columns = [
            "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
            "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
            "Passport", "PitchSatisfactionScore", "OwnCar",
            "NumberOfChildrenVisiting", "MonthlyIncome",
        ]

        self.models: Dict[str, Dict] = {}
        self.best_model_name: Optional[str] = None
        self.best_f1_score: float = -1.0
        self.best_model_threshold: float = 0.5

        self.df_train = pd.DataFrame()
        self.df_test = pd.DataFrame()

        logger.info("Initialized BuildingModels")
        logger.info("HF dataset repo: %s", self.config.ds_repo_id)
        logger.info("HF model repo:   %s", self.config.model_repo_id)
        logger.info("Local model dir: %s", self.model_dir)

    # ---------- Data ----------
    def load_data_from_hf(self) -> bool:
        try:
            train_path = hf_hub_download(
                repo_id=self.config.ds_repo_id,
                filename=self.config.train_file,
                repo_type="dataset",
                token=self.hf_token,
            )
            test_path = hf_hub_download(
                repo_id=self.config.ds_repo_id,
                filename=self.config.test_file,
                repo_type="dataset",
                token=self.hf_token,
            )

            self.df_train = pd.read_csv(train_path)
            self.df_test = pd.read_csv(test_path)

            logger.info("Train shape: %s | Test shape: %s", self.df_train.shape, self.df_test.shape)
            return True

        except HfHubHTTPError as e:
            logger.error("HF HTTP error while downloading datasets: %s", e)
            return False
        except Exception as e:
            logger.exception("Unexpected error while loading data: %s", e)
            return False

    def _split_xy(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        if self.target_col not in self.df_train.columns or self.target_col not in self.df_test.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in train/test.")
        X_train = self.df_train.drop(columns=[self.target_col])
        y_train = self.df_train[self.target_col]
        X_test = self.df_test.drop(columns=[self.target_col])
        y_test = self.df_test[self.target_col]
        return X_train, y_train, X_test, y_test

    # ---------- Preprocess ----------
    def _make_preprocessor(self) -> ColumnTransformer:
        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ])

        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.numerical_columns),
                ("cat", cat_pipe, self.categorical_columns),
            ],
            remainder="drop",
        )

    # ---------- Model building ----------
    def build_models(self) -> Dict[str, Dict]:
        X_train, y_train, _, _ = self._split_xy()
        preprocessor = self._make_preprocessor()

        models_params = {
            "DecisionTreeClassifier": {
                "model": DecisionTreeClassifier(class_weight="balanced", random_state=self.config.random_state),
                "params": {
                    "classifier__criterion": ["gini", "entropy"],
                    "classifier__max_depth": [2, 3, 4, None],
                    "classifier__min_samples_leaf": [1, 2, 4],
                    "classifier__min_samples_split": [2, 5, 10],
                    "classifier__max_features": ["sqrt", "log2", None],
                },
            },
            "RandomForestClassifier": {
                "model": RandomForestClassifier(class_weight="balanced", random_state=self.config.random_state),
                "params": {
                    "classifier__n_estimators": [100, 200, 300],
                    "classifier__max_depth": [5, 10, 15, None],
                    "classifier__min_samples_split": [2, 5, 10],
                    "classifier__min_samples_leaf": [1, 2, 4],
                    "classifier__max_features": [0.3, 0.5, 0.7],
                    "classifier__bootstrap": [True],
                },
            },
            "GradientBoostingClassifier": {
                "model": GradientBoostingClassifier(random_state=self.config.random_state),
                "params": {
                    "classifier__n_estimators": [100, 200, 300],
                    "classifier__learning_rate": [0.01, 0.05, 0.1],
                    "classifier__subsample": [0.7, 0.8, 0.9],
                    "classifier__max_depth": [2, 3, 4],
                    "classifier__min_samples_leaf": [1, 2, 4],
                },
            },
        }

        cv = KFold(n_splits=self.config.cv_splits, shuffle=True, random_state=self.config.random_state)

        for model_name, spec in models_params.items():
            logger.info("Training %s ...", model_name)

            pipe = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", spec["model"]),
            ])

            search = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=spec["params"],
                n_iter=self.config.n_iter_search,
                cv=cv,
                scoring="f1",
                random_state=self.config.random_state,
                n_jobs=-1,
                verbose=0,
            )

            with mlflow.start_run(run_name=model_name):
                search.fit(X_train, y_train)

                best_est = search.best_estimator_
                best_score = float(search.best_score_)
                best_params = search.best_params_

                self.models[model_name] = {
                    "model": best_est,
                    "best_score_cv_f1": best_score,
                    "best_params": best_params,
                }

                # Save locally
                out_path = self.model_dir / f"{model_name}.joblib"
                joblib.dump(best_est, out_path)

                # Log to MLflow
                mlflow.log_params(best_params)
                mlflow.log_metric("cv_best_f1", best_score)
                mlflow.log_artifact(str(out_path), artifact_path="models")

                logger.info("%s done | CV best F1: %.4f", model_name, best_score)

        return self.models

    # ---------- Evaluation ----------
    @staticmethod
    def _best_threshold_by_f1(y_true: pd.Series, y_prob: np.ndarray) -> Tuple[float, float]:
        precision, recall, thresh = precision_recall_curve(y_true, y_prob)
        f1s = 2 * (precision * recall) / (precision + recall + 1e-10)

        # precision_recall_curve returns thresh with length-1 vs precision/recall
        if len(thresh) == 0:
            return 0.5, float(np.nanmax(f1s))

        idx = int(np.nanargmax(f1s[:-1]))  # align with thresh length
        return float(thresh[idx]), float(f1s[idx])

    def evaluate_models(self) -> pd.DataFrame:
        _, _, X_test, y_test = self._split_xy()
        rows = []

        for model_name, info in self.models.items():
            model = info["model"]

            with mlflow.start_run(run_name=f"{model_name}_eval"):
                if not hasattr(model, "predict_proba"):
                    logger.warning("%s has no predict_proba; skipping threshold tuning.", model_name)
                    y_pred = model.predict(X_test)
                    y_prob = None
                    threshold = 0.5
                else:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    threshold, _ = self._best_threshold_by_f1(y_test, y_prob)
                    y_pred = (y_prob >= threshold).astype(int)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                report = classification_report(y_test, y_pred, zero_division=0)
                cm = confusion_matrix(y_test, y_pred)

                # log
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("threshold", threshold)
                mlflow.log_text(report, f"{model_name}_classification_report.txt")

                rows.append({
                    "model": model_name,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "best_threshold": threshold,
                })

                # Track best
                if f1 > self.best_f1_score:
                    self.best_f1_score = f1
                    self.best_model_name = model_name
                    self.best_model_threshold = threshold

                logger.info("%s | F1=%.4f | threshold=%.4f", model_name, f1, threshold)

        return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

    # ---------- Register best model to HF ----------
    def register_best_model_hf(self) -> bool:
        if not self.best_model_name:
            logger.error("No best model selected. Run evaluate_models() first.")
            return False

        best_model = self.models[self.best_model_name]["model"]

        # Save artifacts locally
        best_model_path = self.model_dir / f"BestModel_{self.best_model_name}.joblib"
        threshold_path = self.model_dir / "best_threshold.txt"

        joblib.dump(best_model, best_model_path)
        threshold_path.write_text(str(self.best_model_threshold))

        # Ensure HF model repo exists
        try:
            create_repo(
                repo_id=self.config.model_repo_id,
                repo_type="model",
                private=False,
                exist_ok=True,
                token=self.hf_token,
            )
        except Exception as e:
            logger.exception("Failed to ensure HF model repo exists: %s", e)
            return False

        try:
            logger.info("Uploading best model to HF: %s", self.config.model_repo_id)
            self.api.upload_file(
                path_or_fileobj=str(best_model_path),
                path_in_repo=f"Model_Dump_JOBLIB/{best_model_path.name}",
                repo_id=self.config.model_repo_id,
                repo_type="model",
                token=self.hf_token,
            )
            self.api.upload_file(
                path_or_fileobj=str(threshold_path),
                path_in_repo="Model_Dump_JOBLIB/best_threshold.txt",
                repo_id=self.config.model_repo_id,
                repo_type="model",
                token=self.hf_token,
            )
            logger.info("Best model + threshold uploaded.")
            return True

        except (RepositoryNotFoundError, HfHubHTTPError) as e:
            logger.error("HF error uploading model artifacts: %s", e)
            return False
        except Exception as e:
            logger.exception("Unexpected error uploading model artifacts: %s", e)
            return False

    # ---------- Pipeline ----------
    def run(self) -> bool:
        if not self.load_data_from_hf():
            return False

        self.build_models()
        metrics = self.evaluate_models()
        logger.info("Model metrics:\n%s", metrics)

        if metrics.empty:
            logger.error("No metrics produced.")
            return False

        return self.register_best_model_hf()
