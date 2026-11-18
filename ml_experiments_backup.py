
# Snow-Proof: Machine Learning Experiments
# Test 1: Reproduce our Random Forest from Test 56

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, brier_score_loss


# 1. Load our prepared dataset
ml_data = pd.read_csv(
    "ml_dataset.csv",
    parse_dates=["week_start"]
)

print("ML dataset loaded!")
print("Examples:", len(ml_data))
print("Stations:", ml_data["station_code"].nunique())


# 2. Recreate the original train-validation split
week_end = ml_data["week_start"] + pd.Timedelta(days=6)

train_data = ml_data[
    week_end < pd.Timestamp("2022-01-01")
].copy()

validation_data = ml_data[
    ml_data["week_start"] >= pd.Timestamp("2022-01-01")
].copy()

print("\nTraining examples:", len(train_data))
print("Validation examples:", len(validation_data))


# 3. Use exactly the same features as Test 56
features = [
    "altitude_m",
    "month",
    "day",
    "station_code"
]

X_train = train_data[features]
y_train = train_data["good_week"].astype(int)

X_val = validation_data[features]
y_val = validation_data["good_week"].astype(int)


# 4. Convert station codes into numerical columns
preprocessor = ColumnTransformer(
    transformers=[
        (
            "station",
            OneHotEncoder(handle_unknown="ignore"),
            ["station_code"]
        )
    ],
    remainder="passthrough"
)


# 5. Create the same Random Forest
forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=20,
    random_state=42
)

model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", forest)
])


# 6. Train the model
model.fit(X_train, y_train)


# 7. Evaluate on the validation data
probabilities = model.predict_proba(X_val)[:, 1]
predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)
brier = brier_score_loss(y_val, probabilities)

print("\nTEST: Reproduce original Random Forest")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Brier Score:", round(brier, 4))


# Test 2: Reproduce walk-forward validation (old Test 70)

print("\nTEST 2: Walk-forward validation")

for test_year in [2020, 2021, 2022, 2023]:

    year_start = pd.Timestamp(f"{test_year}-01-01")
    next_year = pd.Timestamp(f"{test_year + 1}-01-01")

    # Train using only weeks ending before the test year
    train = ml_data[
        ml_data["week_start"] + pd.Timedelta(days=6) < year_start
    ].copy()

    # Test on weeks fully contained in the test year
    test = ml_data[
        (ml_data["week_start"] >= year_start) &
        (ml_data["week_start"] + pd.Timedelta(days=6) < next_year)
    ].copy()

    # Only use stations already available during training
    test = test[
        test["station_code"].isin(train["station_code"].unique())
    ].copy()

    X_train_year = train[features]
    y_train_year = train["good_week"].astype(int)

    X_test_year = test[features]
    y_test_year = test["good_week"].astype(int)

    # New model with exactly the same settings
    preprocessing_year = ColumnTransformer(
        transformers=[
            (
                "station",
                OneHotEncoder(handle_unknown="ignore"),
                ["station_code"]
            )
        ],
        remainder="passthrough"
    )

    forest_year = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42
    )

    model_year = Pipeline([
        ("preprocessing", preprocessing_year),
        ("model", forest_year)
    ])

    model_year.fit(X_train_year, y_train_year)

    probabilities_year = model_year.predict_proba(
        X_test_year
    )[:, 1]

    predictions_year = (probabilities_year >= 0.5).astype(int)

    print("\nYear:", test_year)
    print("Training examples:", len(train))
    print("Test examples:", len(test))
    print(
        "RF Accuracy:",
        round(accuracy_score(y_test_year, predictions_year) * 100, 2),
        "%"
    )
    print(
        "RF Brier:",
        round(brier_score_loss(y_test_year, probabilities_year), 4)
    )


# Test 3: Reproduce station baseline from old Test 70

print("\nTEST 3: Historical station baseline")

for test_year in [2020, 2021, 2022, 2023]:

    year_start = pd.Timestamp(f"{test_year}-01-01")
    next_year = pd.Timestamp(f"{test_year + 1}-01-01")

    # Use the same time periods as Test 2
    train = ml_data[
        ml_data["week_start"] + pd.Timedelta(days=6) < year_start
    ].copy()

    test = ml_data[
        (ml_data["week_start"] >= year_start) &
        (ml_data["week_start"] + pd.Timedelta(days=6) < next_year)
    ].copy()

    test = test[
        test["station_code"].isin(train["station_code"].unique())
    ].copy()

    # Calculate historical probability for each station
    station_history = train.groupby(
        "station_code"
    )["good_week"].mean()

    # Assign probabilities to the test examples
    baseline_probabilities = test["station_code"].map(
        station_history
    )

    # Compare probabilities with actual results
    baseline_brier = brier_score_loss(
        test["good_week"].astype(int),
        baseline_probabilities
    )

    print(
        "Year:", test_year,
        "| Baseline Brier:", round(baseline_brier, 4)
    )


# Test 4: Reproduce monthly baseline from old Test 71

print("\nTEST 4: Historical monthly baseline")

for test_year in [2020, 2021, 2022, 2023]:

    year_start = pd.Timestamp(f"{test_year}-01-01")
    next_year = pd.Timestamp(f"{test_year + 1}-01-01")

    # Training data: only previous years
    train = ml_data[
        ml_data["week_start"] + pd.Timedelta(days=6) < year_start
    ].copy()

    # Test data: weeks fully within the selected year
    test = ml_data[
        (ml_data["week_start"] >= year_start) &
        (ml_data["week_start"] + pd.Timedelta(days=6) < next_year)
    ].copy()

    # Only stations already present in training
    test = test[
        test["station_code"].isin(train["station_code"].unique())
    ].copy()

    # Historical probability per station and month
    monthly_history = train.groupby(
        ["station_code", "month"]
    )["good_week"].mean()

    # Fallback if a station-month combination is unavailable
    station_history = train.groupby(
        "station_code"
    )["good_week"].mean()

    monthly_probabilities = [
        monthly_history.get(
            (row.station_code, row.month),
            station_history[row.station_code]
        )
        for row in test.itertuples()
    ]

    monthly_brier = brier_score_loss(
        test["good_week"].astype(int),
        monthly_probabilities
    )

    print(
        "Year:", test_year,
        "| Monthly Baseline Brier:",
        round(monthly_brier, 4)
    )


# Test 5: Reproduce rolling 14-day baseline from old Test 72

print("\nTEST 5: Rolling 14-day baseline")

import numpy as np

for test_year in [2020, 2021, 2022, 2023]:

    year_start = pd.Timestamp(f"{test_year}-01-01")
    next_year = pd.Timestamp(f"{test_year + 1}-01-01")

    # Training data: only previous years
    train = ml_data[
        ml_data["week_start"] + pd.Timedelta(days=6) < year_start
    ].copy()

    # Test data: weeks fully within the selected year
    test = ml_data[
        (ml_data["week_start"] >= year_start) &
        (ml_data["week_start"] + pd.Timedelta(days=6) < next_year)
    ].copy()

    test = test[
        test["station_code"].isin(train["station_code"].unique())
    ].copy()

    # Convert dates to positions in a reference year
    for dataset in [train, test]:
        dataset["calendar_day"] = pd.to_datetime(
            "2000-" +
            dataset["month"].astype(str).str.zfill(2) +
            "-" +
            dataset["day"].astype(str).str.zfill(2)
        ).dt.dayofyear

    # Historical probability per station as fallback
    station_history = train.groupby(
        "station_code"
    )["good_week"].mean()

    rolling_prob = pd.Series(index=test.index, dtype=float)

    # Find historically comparable dates within +/- 14 days
    for code, station_test in test.groupby("station_code"):

        station_train = train[
            train["station_code"] == code
        ]

        for day, rows in station_test.groupby("calendar_day"):

            distance = abs(
                station_train["calendar_day"] - day
            )

            # Correctly handle the change from December to January
            distance = np.minimum(distance, 366 - distance)

            nearby = station_train[distance <= 14]

            if len(nearby) > 0:
                probability = nearby["good_week"].mean()
            else:
                probability = station_history[code]

            rolling_prob.loc[rows.index] = probability

    rolling_brier = brier_score_loss(
        test["good_week"].astype(int),
        rolling_prob
    )

    print(
        "Year:", test_year,
        "| Rolling Baseline Brier:",
        round(rolling_brier, 4)
    )
