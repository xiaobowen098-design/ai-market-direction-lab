"""Train and evaluate a market-direction model."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

FEATURE_COLUMNS = [
    "return_1d",
    "ma_5_vs_20",
    "rsi_14",
    "macd",
    "macd_signal",
    "volatility_20",
    "volume_vs_20d_avg",
]


def train_and_evaluate(data: pd.DataFrame):
    """Train on earlier dates and evaluate on the latest 20% of dates."""
    clean_data = data.replace([np.inf, -np.inf], np.nan)
    clean_data = clean_data.dropna(
        subset=FEATURE_COLUMNS + ["target_up_5d"]
    ).copy()

    if len(clean_data) < 100:
        raise ValueError("At least 100 rows of feature data are required.")

    split_at = int(len(clean_data) * 0.8)

    # Leave a five-day gap because the target looks five trading days ahead.
    train_data = clean_data.iloc[: split_at - 5]
    test_data = clean_data.iloc[split_at:]

    x_train = train_data[FEATURE_COLUMNS]
    y_train = train_data["target_up_5d"].astype(int)
    x_test = test_data[FEATURE_COLUMNS]
    y_test = test_data["target_up_5d"].astype(int)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x_train, y_train)

    predicted = model.predict(x_test)
    up_class_index = list(model.classes_).index(1)
    probability_up = model.predict_proba(x_test)[:, up_class_index]

    metrics = {
        "accuracy": accuracy_score(y_test, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
        "precision": precision_score(y_test, predicted, zero_division=0),
        "recall": recall_score(y_test, predicted, zero_division=0),
    }

    if y_test.nunique() == 2:
        metrics["roc_auc"] = roc_auc_score(y_test, probability_up)

    predictions = pd.DataFrame(
        {
            "actual_up": y_test,
            "predicted_up": predicted,
            "probability_up": probability_up,
        },
        index=test_data.index,
    )

    return model, metrics, predictions
