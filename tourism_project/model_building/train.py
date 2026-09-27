"""Stage 3: train, tune, and evaluate the models, track them in MLflow, and register the winner."""
import json
import os
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
import xgboost
from huggingface_hub import HfApi, create_repo, hf_hub_download
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from common import (ARTIFACT_DIR, CATEGORICAL_FEATURES, DATASET_REPO, FEATURES, HF_TOKEN, META_FILE,
                    MODEL_FILE, MODEL_REPO, NUMERIC_FEATURES, PROCESSED_DIR, TARGET, USE_HUB)

SEED = 42
MIN_TEST_F1 = float(os.getenv("MIN_TEST_F1", "0.60"))
MIN_TEST_RECALL = float(os.getenv("MIN_TEST_RECALL", "0.65"))
N_ITER = int(os.getenv("N_SEARCH_ITER", "25"))
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1",
           "recall": "recall", "precision": "precision"}


def load_split(name):
    if USE_HUB:
        path = hf_hub_download(DATASET_REPO, f"processed/{name}.csv", repo_type="dataset", token=HF_TOKEN)
    else:
        path = PROCESSED_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    return df[FEATURES], df[TARGET].astype(int)


def make_pipeline(model):
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                   min_frequency=10, sparse_output=False))]),
         CATEGORICAL_FEATURES),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def evaluate(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred), "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred), "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba), "pr_auc": average_precision_score(y_true, proba),
    }


def best_f1_threshold(y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    idx = int(np.nanargmax(f1[:-1]))
    return float(thr[idx])


def model_card(meta):
    m = meta["test_metrics"]
    return f"""---
license: mit
tags: [tabular-classification, scikit-learn, xgboost, mlops]
datasets: [{DATASET_REPO}]
---
# Wellness Tourism Package: purchase propensity model

Predicts whether a "Visit with Us" customer will buy the Wellness Tourism Package (`ProdTaken`).

* **Algorithm:** {meta['model_name']} inside a scikit-learn Pipeline (impute + scale + one-hot)
* **Decision threshold:** {meta['threshold']:.3f} (maximises out-of-fold F1)
* **Trained:** {meta['trained_at']}  |  commit `{meta['git_sha']}`

| Test metric | Value |
|---|---|
| Accuracy | {m['accuracy']:.3f} |
| Precision | {m['precision']:.3f} |
| Recall | {m['recall']:.3f} |
| F1 | {m['f1']:.3f} |
| ROC-AUC | {m['roc_auc']:.3f} |
| PR-AUC | {m['pr_auc']:.3f} |

```python
import joblib, json
from huggingface_hub import hf_hub_download
model = joblib.load(hf_hub_download("{MODEL_REPO}", "{MODEL_FILE}"))
meta = json.load(open(hf_hub_download("{MODEL_REPO}", "{META_FILE}")))
proba = model.predict_proba(df)[:, 1]
buy = proba >= meta["threshold"]
```
"""


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ARTIFACT_DIR / 'mlflow.db'}"))
    mlflow.set_experiment("wellness-tourism-purchase")

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"train={X_train.shape} test={X_test.shape} scale_pos_weight={spw:.2f}")

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                random_state=SEED, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
        "xgboost": XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5,
                                 scale_pos_weight=spw, eval_metric="logloss",
                                 random_state=SEED, n_jobs=-1),
    }
    search_spaces = {
        "random_forest": {
            "model__n_estimators": randint(200, 600), "model__max_depth": [None, 8, 12, 16, 24],
            "model__min_samples_leaf": randint(1, 6), "model__max_features": ["sqrt", 0.5, 0.7],
        },
        "xgboost": {
            "model__n_estimators": randint(200, 700), "model__max_depth": randint(3, 10),
            "model__learning_rate": loguniform(0.02, 0.3), "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.5, 0.5), "model__min_child_weight": randint(1, 8),
            "model__gamma": uniform(0, 2), "model__reg_lambda": loguniform(0.1, 10),
        },
    }

    with mlflow.start_run(run_name=f"pipeline-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"):
        mlflow.log_params({"n_train": len(X_train), "n_test": len(X_test), "cv_folds": 5,
                           "positive_rate": round(float(y_train.mean()), 4)})

        # 1) Baseline comparison
        leaderboard = []
        for name, model in candidates.items():
            with mlflow.start_run(run_name=f"baseline-{name}", nested=True):
                cv = cross_validate(make_pipeline(model), X_train, y_train, cv=CV, scoring=SCORING, n_jobs=-1)
                row = {"model": name, **{k: cv[f"test_{k}"].mean() for k in SCORING}}
                mlflow.log_param("stage", "baseline")
                mlflow.log_metrics({f"cv_{k}": v for k, v in row.items() if k != "model"})
                leaderboard.append(row)
        board = pd.DataFrame(leaderboard).sort_values("pr_auc", ascending=False)
        print("\nBaseline 5-fold CV leaderboard:\n", board.round(4).to_string(index=False))
        board.to_csv(ARTIFACT_DIR / "baseline_leaderboard.csv", index=False)
        mlflow.log_artifact(str(ARTIFACT_DIR / "baseline_leaderboard.csv"))

        # 2) Hyper-parameter tuning of the tree ensembles
        tuned = {}
        for name, space in search_spaces.items():
            with mlflow.start_run(run_name=f"tuned-{name}", nested=True):
                search = RandomizedSearchCV(make_pipeline(candidates[name]), space, n_iter=N_ITER,
                                            scoring="average_precision", cv=CV, random_state=SEED,
                                            n_jobs=-1, refit=True)
                search.fit(X_train, y_train)
                tuned[name] = search
                mlflow.log_param("stage", "tuned")
                mlflow.log_params({k.replace("model__", ""): v for k, v in search.best_params_.items()})
                mlflow.log_metric("cv_pr_auc", search.best_score_)
                print(f"Tuned {name:14s} CV PR-AUC = {search.best_score_:.4f}")

        best_name = max(tuned, key=lambda n: tuned[n].best_score_)
        best = tuned[best_name].best_estimator_
        print(f"\nSelected model: {best_name}")

        # 3) Threshold from out-of-fold predictions on the training set only
        oof = cross_val_predict(best, X_train, y_train, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
        threshold = best_f1_threshold(y_train, oof)
        print(f"Chosen decision threshold (max OOF F1): {threshold:.3f}")

        # 4) One final evaluation on the untouched test set
        best.fit(X_train, y_train)
        test_proba = best.predict_proba(X_test)[:, 1]
        train_metrics = evaluate(y_train, best.predict_proba(X_train)[:, 1], threshold)
        oof_metrics = evaluate(y_train, oof, threshold)        # honest estimate of generalisation
        test_metrics = evaluate(y_test, test_proba, threshold)
        default_metrics = evaluate(y_test, test_proba, 0.5)
        print("\nTest classification report @ tuned threshold:\n",
              classification_report(y_test, (test_proba >= threshold).astype(int), digits=3))
        print("Confusion matrix:\n", confusion_matrix(y_test, (test_proba >= threshold).astype(int)))
        print("\nTrain vs test (tuned threshold):")
        print(pd.DataFrame({"train (fit)": train_metrics, "train (5-fold OOF)": oof_metrics,
                            "test": test_metrics, "test@0.5": default_metrics}).round(4).to_string())

        mlflow.log_param("selected_model", best_name)
        mlflow.log_metric("threshold", threshold)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"oof_{k}": v for k, v in oof_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # 5) Persist the artefacts
        meta = {
            "model_name": best_name, "threshold": threshold, "features": FEATURES,
            "numeric_features": NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES,
            "best_params": {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                            for k, v in tuned[best_name].best_params_.items()},
            "train_metrics": train_metrics, "oof_metrics": oof_metrics, "test_metrics": test_metrics,
            "test_metrics_at_0_5": default_metrics,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": os.getenv("GITHUB_SHA", "local")[:7],
            "versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
        }
        joblib.dump(best, ARTIFACT_DIR / MODEL_FILE)
        (ARTIFACT_DIR / META_FILE).write_text(json.dumps(meta, indent=2))
        (ARTIFACT_DIR / "README.md").write_text(model_card(meta))
        pd.DataFrame({"y_true": y_test.values, "proba": test_proba}).to_csv(
            ARTIFACT_DIR / "test_predictions.csv", index=False)
        mlflow.log_artifacts(str(ARTIFACT_DIR), artifact_path="model_bundle")

        # 6) Quality gate: never ship a model that regressed
        passed = test_metrics["f1"] >= MIN_TEST_F1 and test_metrics["recall"] >= MIN_TEST_RECALL
        mlflow.set_tag("quality_gate", "passed" if passed else "failed")
        if not passed:
            raise SystemExit(f"Quality gate FAILED: f1={test_metrics['f1']:.3f} (min {MIN_TEST_F1}), "
                             f"recall={test_metrics['recall']:.3f} (min {MIN_TEST_RECALL})")
        print(f"\nQuality gate passed (F1 >= {MIN_TEST_F1}, recall >= {MIN_TEST_RECALL}).")

        if USE_HUB:
            create_repo(MODEL_REPO, repo_type="model", token=HF_TOKEN, exist_ok=True, private=False)
            api = HfApi(token=HF_TOKEN)
            for fname in (MODEL_FILE, META_FILE, "README.md"):
                api.upload_file(path_or_fileobj=str(ARTIFACT_DIR / fname), path_in_repo=fname,
                                repo_id=MODEL_REPO, repo_type="model",
                                commit_message=f"{best_name} f1={test_metrics['f1']:.3f}")
            print(f"Registered model -> https://huggingface.co/{MODEL_REPO}")
        else:
            print(f"[local mode] Model saved to {ARTIFACT_DIR / MODEL_FILE}")


if __name__ == "__main__":
    main()
