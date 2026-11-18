"""Gezielter Modellvergleich: python run_ml_comparison.py"""

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss

from snowproof_ml import (
    FEATURES, create_random_forest, monthly_baseline,
    prepare_dataset, split_dataset, station_baseline,
)


def report(name, actual, probabilities):
    probabilities = np.asarray(probabilities)
    prediction = (probabilities >= 0.5).astype(int)
    print(f"{name:<23} Accuracy: {accuracy_score(actual, prediction) * 100:6.2f}%"
          f" | Brier: {brier_score_loss(actual, probabilities):.4f}")


def main():
    data = prepare_dataset()
    train, val = split_dataset(data)
    print(f"Training: {len(train)} Wochen | Validierung: {len(val)} Wochen")
    print(f"Validierungsstationen: {val['station_code'].nunique()}")

    model = create_random_forest()
    model.fit(train[FEATURES], train["good_week"].astype(int))
    prob_rf = model.predict_proba(val[FEATURES])[:, 1]
    actual = val["good_week"].astype(int)

    print("\n--- Vergleich 2022-2023 ---")
    report("RF mit Stationsidentität", actual, prob_rf)
    report("Stations-Baseline", actual, station_baseline(train, val))
    report("Monats-Baseline", actual, monthly_baseline(train, val))
    print("\n2024 bleibt für den abschliessenden unabhängigen Test reserviert.")


if __name__ == "__main__":
    main()
