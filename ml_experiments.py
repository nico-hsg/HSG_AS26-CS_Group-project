
# Snow-Proof: Machine Learning Experiments
# Test 1: Reproduce our Random Forest from Test 56

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, brier_score_loss

def split_data_by_year(dataset, test_year):

    year_start = pd.Timestamp(f"{test_year}-01-01")
    next_year = pd.Timestamp(f"{test_year + 1}-01-01")

    # Training: only weeks ending before test year
    train = dataset[
        dataset["week_start"] + pd.Timedelta(days=6) < year_start
    ].copy()

    # Testing: only weeks fully within test year
    test = dataset[
        (dataset["week_start"] >= year_start) &
        (dataset["week_start"] + pd.Timedelta(days=6) < next_year)
    ].copy()

    # Only test stations already available during training
    test = test[
        test["station_code"].isin(train["station_code"].unique())
    ].copy()

    return train, test
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

    # Use our shared train-test split function
    train, test = split_data_by_year(ml_data, test_year)

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

    # Compare rolling baseline performance by winter month
    for month in [12, 1, 2, 3]:

        month_mask = test["month"] == month

        if not month_mask.any():
            continue

        month_brier = brier_score_loss(
            test.loc[month_mask, "good_week"].astype(int),
            rolling_prob.loc[month_mask]
        )

        print(
            "Year:", test_year,
            "| Month:", month,
            "| Baseline Brier:", round(month_brier, 4)
        )


# Test 6: Shared train-test split function

print("\nTEST 6: Shared train-test split")




# Check whether our function reproduces the original splits
expected_counts = {
    2020: (12749, 2447),
    2021: (15299, 2469),
    2022: (17870, 2465),
    2023: (20443, 2496)
}

for year, expected in expected_counts.items():

    train_check, test_check = split_data_by_year(
        ml_data, year
    )

    actual = (len(train_check), len(test_check))

    print(
        "Year:", year,
        "| Train:", len(train_check),
        "| Test:", len(test_check),
        "| Match:", actual == expected
    )

    assert actual == expected, (
        f"Unexpected train/test counts for {year}"
    )


# Test 7: Random Forest performance by winter month

print("\nTEST 7: Random Forest by winter month")

winter_results = []

for test_year in [2020, 2021, 2022, 2023]:

    train, test = split_data_by_year(ml_data, test_year)

    # Use the same Random Forest settings as before
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "station",
                OneHotEncoder(handle_unknown="ignore"),
                ["station_code"]
            )
        ],
        remainder="passthrough"
    )

    forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42
    )

    winter_model = Pipeline([
        ("preprocessing", preprocessing),
        ("model", forest)
    ])

    winter_model.fit(
        train[features],
        train["good_week"].astype(int)
    )

    # Predict probabilities for the test year
    test = test.copy()
    test["rf_probability"] = winter_model.predict_proba(
        test[features]
    )[:, 1]

    # Evaluate each winter month separately
    for month in [12, 1, 2, 3]:

        month_data = test[test["month"] == month]

        if len(month_data) == 0:
            continue

        month_brier = brier_score_loss(
            month_data["good_week"].astype(int),
            month_data["rf_probability"]
        )

        winter_results.append({
            "year": test_year,
            "month": month,
            "examples": len(month_data),
            "brier": month_brier
        })

        print(
            "Year:", test_year,
            "| Month:", month,
            "| Examples:", len(month_data),
            "| RF Brier:", round(month_brier, 4)
        )


# Test 8: First prediction for a selected ski week

print("\nTEST 8: First prediction for a selected ski week")

# Select Davos from our existing training data
davos = train_data[
    train_data["station_code"] == "DAV"
]

# Get the altitude of the Davos measurement station
altitude = davos["altitude_m"].iloc[0]

# Prepare the four features for 1 February 2027
prediction_input = pd.DataFrame([{
    "altitude_m": altitude,
    "month": 2,
    "day": 1,
    "station_code": "DAV"
}])

# Use our already trained Random Forest
probability = model.predict_proba(prediction_input)[0, 1]

print("Station: DAV")
print("Week: 2027-02-01 to 2027-02-07")
print("Predicted probability:", round(probability * 100, 2), "%")



# Test 9: Prediction function

print("\nTEST 9: Prediction function")


def predict_good_week(model, resort, start_date, end_date):

    # Convert dates to Pandas timestamps
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Check whether the selected period has exactly 7 days
    if (end - start).days != 6:
        return None

    # Check whether all days are within the ski season
    week_dates = pd.date_range(start, end)

    if not all(day.month in [12, 1, 2, 3, 4] for day in week_dates):
        return None

    # Find the measurement station
    station_code = resort.get("station_code")

    if pd.isna(station_code):
        return None

    # Check whether the station was part of training
    if station_code not in model.snowproof_station_codes:
        return None

    # Get the station altitude from the trained model
    altitude = model.snowproof_station_altitudes[station_code]

    # Prepare the same features as during training
    prediction_input = pd.DataFrame([{
        "altitude_m": altitude,
        "month": start.month,
        "day": start.day,
        "station_code": station_code
    }])

    # Calculate probability
    probability = model.predict_proba(prediction_input)[0, 1]

    return float(probability)


# Test 10: Check our prediction function

print("\nTEST 10: Check prediction function")

# Store the stations used during training
model.snowproof_station_codes = set(
    train_data["station_code"]
)

# Store their altitudes
model.snowproof_station_altitudes = dict(
    zip(train_data["station_code"], train_data["altitude_m"])
)

# Create a sample resort connected to Davos station
test_resort = {
    "station_code": "DAV"
}

# Call our prediction function
result = predict_good_week(
    model,
    test_resort,
    "2027-02-01",
    "2027-02-07"
)

print("Prediction:", round(result * 100, 2), "%")

# Check whether the result is a valid probability
assert 0 <= result <= 1

# Compare with our manual prediction from Test 8
assert abs(result - float(probability)) < 0.000001

print("Test 10 passed!")


# Test 11: Check invalid prediction inputs

print("\nTEST 11: Invalid prediction inputs")

# Test 1: Only five days instead of seven
result_1 = predict_good_week(
    model, test_resort, "2027-02-01", "2027-02-05"
)

# Test 2: Week outside the ski season
result_2 = predict_good_week(
    model, test_resort, "2027-07-01", "2027-07-07"
)

# Test 3: Resort without a measurement station
result_3 = predict_good_week(
    model, {"station_code": None},
    "2027-02-01", "2027-02-07"
)

# Test 4: Station not included in training
result_4 = predict_good_week(
    model, {"station_code": "UNKNOWN"},
    "2027-02-01", "2027-02-07"
)

print("Invalid duration:", result_1)
print("Outside ski season:", result_2)
print("Missing station:", result_3)
print("Unknown station:", result_4)

assert result_1 is None
assert result_2 is None
assert result_3 is None
assert result_4 is None

print("Test 11 passed!")


# Test 12: Predictions for real ski resorts

print("\nTEST 12: Predictions for real ski resorts")

from data import load_resorts, load_stations, link_stations_to_resorts

# Load real resorts and measurement stations
resorts = load_resorts()
stations = load_stations()

# Connect each resort to a suitable measurement station
linked_resorts = link_stations_to_resorts(resorts, stations)

# Test three resorts with supported stations
supported_resorts = linked_resorts[
    linked_resorts["station_code"].isin(
        model.snowproof_station_codes
    )
].head(3)

assert len(supported_resorts) == 3

# Predict the same holiday week for each resort
for _, resort in supported_resorts.iterrows():

    result = predict_good_week(
        model,
        resort,
        "2027-02-01",
        "2027-02-07"
    )

    print("Resort:", resort["name"])
    print("Station:", resort["station_code"])
    print("Probability:", round(result * 100, 2), "%")
    print("------------------------")

    assert 0 <= result <= 1

print("Test 12 passed!")


# Test 13: Predictions for multiple ski resorts

print("\nTEST 13: Predictions for multiple resorts")


def predict_resorts(model, resorts, start_date, end_date):

    results = []

    # Go through each resort
    for _, resort in resorts.iterrows():

        # Use our already tested prediction function
        probability = predict_good_week(
            model, resort, start_date, end_date
        )

        # Store the result for Jan's ranking
        results.append({
            "resort_id": resort["resort_id"],
            "week_start": str(pd.to_datetime(start_date).date()),
            "week_end": str(pd.to_datetime(end_date).date()),
            "probability": probability
        })

    return results


# Test our new function with real resorts
test_resorts = linked_resorts.head(5)

ranking_predictions = predict_resorts(
    model,
    test_resorts,
    "2027-02-01",
    "2027-02-07"
)

# Show the predictions
for prediction in ranking_predictions:
    print(prediction)

# Verify the result structure
assert len(ranking_predictions) == len(test_resorts)

for prediction in ranking_predictions:

    assert set(prediction.keys()) == {
        "resort_id",
        "week_start",
        "week_end",
        "probability"
    }

    assert (
        prediction["probability"] is None
        or 0 <= prediction["probability"] <= 1
    )

print("Test 13 passed!")


# Test 14: Compatibility with Jan's ranking

print("\nTEST 14: Ranking compatibility")

from streamlit_app import rank_resorts

# Use all real ski resorts
all_predictions = predict_resorts(
    model,
    linked_resorts,
    "2027-02-01",
    "2027-02-07"
)

# Let Jan's existing function rank the resorts
ranked_resorts = rank_resorts(
    linked_resorts,
    all_predictions
)

# Show the ranking
print(
    ranked_resorts[
        ["resort_id", "name", "station_code", "probability"]
    ].to_string(index=False)
)

# Check that unsupported resorts were removed
assert ranked_resorts["probability"].notna().all()

# Check that the probabilities are sorted highest first
assert ranked_resorts["probability"].is_monotonic_decreasing

# Check that all supported predictions are included
valid_predictions = sum(
    prediction["probability"] is not None
    for prediction in all_predictions
)

assert len(ranked_resorts) == valid_predictions

print("Total resorts:", len(linked_resorts))
print("Ranked resorts:", len(ranked_resorts))
print("Test 14 passed!")


# Test 15: Train the app model using only historical data

print("\nTEST 15: Train app model with historical data")

from streamlit_app import train_model

# Use the existing split for test year 2022
train_2022, test_2022 = split_data_by_year(
    ml_data, 2022
)

# Prepare the labels expected by train_model()
historical_labels = train_2022[
    ["station_code", "week_start", "year", "good_week"]
].copy()

# Prepare station information
station_information = ml_data[
    ["station_code", "altitude_m"]
].drop_duplicates()

# Train the real app model on historical data only
historical_model = train_model(
    historical_labels,
    station_information
)

# Check that the training period is correct
latest_training_date = (
    train_2022["week_start"] + pd.Timedelta(days=6)
).max()

print("Latest training week ends:", latest_training_date)
print("Training examples:", len(train_2022))
print("Test examples:", len(test_2022))

assert latest_training_date < pd.Timestamp("2022-01-01")
assert len(train_2022) == 17870
assert len(test_2022) == 2465

print("Test 15 passed!")


# Test 16: Evaluate the app model on 2022

print("\nTEST 16: Evaluate app model on 2022")

# Prepare the test inputs using the same four features
X_test_2022 = test_2022[
    ["altitude_m", "month", "day", "station_code"]
]

# Get the actual outcomes
y_test_2022 = test_2022["good_week"].astype(int)

# Predict probabilities using the app model from Test 15
app_probabilities = historical_model.predict_proba(
    X_test_2022
)[:, 1]

# Convert probabilities into binary predictions
app_predictions = (app_probabilities >= 0.5).astype(int)

# Calculate evaluation metrics
app_accuracy = accuracy_score(
    y_test_2022, app_predictions
)

app_brier = brier_score_loss(
    y_test_2022, app_probabilities
)

print("App Accuracy:", round(app_accuracy * 100, 2), "%")
print("App Brier:", round(app_brier, 4))

# Check whether the results match our earlier experiments
assert round(app_accuracy * 100, 2) == 89.94
assert round(app_brier, 4) == 0.1049

print("Test 16 passed!")


# Test 17: Evaluate app model for all walk-forward years

print("\nTEST 17: App model walk-forward evaluation")

app_results = []

# Reference scores from our earlier experiments
expected_brier = {
    2020: 0.1141,
    2021: 0.1739,
    2022: 0.1049,
    2023: 0.1510
}

expected_accuracy = {
    2020: 88.43,
    2021: 74.24,
    2022: 89.94,
    2023: 75.24
}

for year in [2020, 2021, 2022, 2023]:

    # Use only historical data for training
    train, test = split_data_by_year(ml_data, year)

    # Train the same model as in our Streamlit app
    year_model = train_model(
        train[["station_code", "week_start", "year", "good_week"]],
        station_information
    )

    # Prepare the test features
    X_test = test[
        ["altitude_m", "month", "day", "station_code"]
    ]

    y_test = test["good_week"].astype(int)

    # Calculate the model probabilities
    probabilities = year_model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(
        y_test, (probabilities >= 0.5).astype(int)
    )

    brier = brier_score_loss(y_test, probabilities)

    # Store the results
    app_results.append({
        "year": year,
        "accuracy": accuracy,
        "brier": brier
    })

    print(
        "Year:", year,
        "| Accuracy:", round(accuracy * 100, 2), "%",
        "| Brier:", round(brier, 4)
    )

    # Verify against our original experiments
    assert round(accuracy * 100, 2) == expected_accuracy[year]
    assert round(brier, 4) == expected_brier[year]

print("Test 17 passed!")


# Test 18: Reusable rolling 14-day baseline

print("\nTEST 18: Rolling baseline function")

def rolling_baseline(train, test):

    train = train.copy()
    test = test.copy()

    # Convert month and day to a comparable calendar position
    for dataset in [train, test]:
        dataset["calendar_day"] = pd.to_datetime(
            "2000-" +
            dataset["month"].astype(str).str.zfill(2) +
            "-" +
            dataset["day"].astype(str).str.zfill(2)
        ).dt.dayofyear

    # Fallback: historical average for each station
    station_history = train.groupby(
        "station_code"
    )["good_week"].mean()

    probabilities = pd.Series(index=test.index, dtype=float)

    # Calculate predictions for each station
    for code, station_test in test.groupby("station_code"):

        station_train = train[
            train["station_code"] == code
        ]

        for day, rows in station_test.groupby("calendar_day"):

            distance = abs(
                station_train["calendar_day"] - day
            )

            # Include dates across the December-January boundary
            distance = np.minimum(distance, 366 - distance)

            nearby = station_train[distance <= 14]

            if len(nearby) > 0:
                probability = nearby["good_week"].mean()
            else:
                probability = station_history[code]

            probabilities.loc[rows.index] = probability

    return probabilities


# Use our already verified 2022 split
baseline_predictions = rolling_baseline(
    train_2022,
    test_2022
)

baseline_brier = brier_score_loss(
    test_2022["good_week"].astype(int),
    baseline_predictions
)

print("Baseline Brier 2022:", round(baseline_brier, 4))

assert round(baseline_brier, 4) == 0.0784
assert baseline_predictions.notna().all()

print("Test 18 passed!")


# Test 19: Rolling baseline for all walk-forward years

print("\nTEST 19: Rolling baseline walk-forward evaluation")

expected_baseline = {
    2020: 0.1103,
    2021: 0.1510,
    2022: 0.0784,
    2023: 0.1921
}

baseline_results = []

for year in [2020, 2021, 2022, 2023]:

    # Use the same historical train/test split as the RF
    train, test = split_data_by_year(ml_data, year)

    # Calculate historical baseline probabilities
    probabilities = rolling_baseline(train, test)

    # Evaluate the probabilities
    brier = brier_score_loss(
        test["good_week"].astype(int),
        probabilities
    )

    baseline_results.append({
        "year": year,
        "brier": brier
    })

    print("Year:", year, "| Baseline Brier:", round(brier, 4))

    # Compare with our original Test 5
    assert round(brier, 4) == expected_baseline[year]
    assert probabilities.notna().all()

print("Test 19 passed!")



# Test 27: Compare different training periods

print("\nTEST 27: Compare training periods for 2022")

from sklearn.base import clone

# Load our extended historical dataset
ml_extended = pd.read_csv(
    "ml_dataset_extended.csv",
    parse_dates=["week_start"]
)

# Use the same 2022 test examples for every model
original_train, test_2022_comparison = split_data_by_year(
    ml_data, 2022
)

X_test = test_2022_comparison[features]
y_test = test_2022_comparison["good_week"].astype(int)

for start_year in [2015, 2010, 2000]:

    # Reproduce the original training dataset
    if start_year == 2015:
        train = original_train.copy()

    else:
        # Select additional historical weeks
        older_train = ml_extended[
            (ml_extended["week_start"] >= pd.Timestamp(
                f"{start_year}-01-01"
            )) &
            (ml_extended["week_start"] < pd.Timestamp(
                "2015-01-01"
            ))
        ].copy()

        # Keep original training rows in their original order
        train = pd.concat(
            [original_train, older_train],
            ignore_index=True
        )

    # Create a fresh Random Forest with the same settings
    candidate = clone(model)

    # Train on the selected historical period
    candidate.fit(
        train[features],
        train["good_week"].astype(int)
    )

    # Predict the same 2022 test examples
    probabilities = candidate.predict_proba(X_test)[:, 1]

    # Calculate accuracy and Brier Score
    accuracy = accuracy_score(
        y_test,
        (probabilities >= 0.5).astype(int)
    )

    brier = brier_score_loss(
        y_test,
        probabilities
    )

    print(
        "Training from:", start_year,
        "| Examples:", len(train),
        "| Accuracy:", round(accuracy * 100, 2), "%",
        "| Brier:", round(brier, 4)
    )

    # Verify the original model
    if start_year == 2015:
        assert round(accuracy * 100, 2) == 89.94
        assert round(brier, 4) == 0.1049

print("Test 27 passed!")


# Test 28: Compare training periods across all test years

print("\nTEST 28: Training periods across 2020-2023")

results_28 = []

for test_year in [2020, 2021, 2022, 2023]:

    # Original chronological train/test split
    original_train, test = split_data_by_year(
        ml_data, test_year
    )

    X_test = test[features]
    y_test = test["good_week"].astype(int)

    for start_year in [2015, 2010, 2000]:

        # Preserve the original training rows and their order
        if start_year == 2015:
            train = original_train.copy()

        else:
            older_train = ml_extended[
                (ml_extended["week_start"] >= pd.Timestamp(
                    f"{start_year}-01-01"
                )) &
                (ml_extended["week_start"] < pd.Timestamp(
                    "2015-01-01"
                ))
            ].copy()

            train = pd.concat(
                [original_train, older_train],
                ignore_index=True
            )

        # Prevent training on weeks overlapping the test year
        assert (
            train["week_start"] + pd.Timedelta(days=6)
            < pd.Timestamp(f"{test_year}-01-01")
        ).all()

        # Fresh Random Forest with unchanged settings
        candidate = clone(model)

        candidate.fit(
            train[features],
            train["good_week"].astype(int)
        )

        probabilities = candidate.predict_proba(
            X_test
        )[:, 1]

        accuracy = accuracy_score(
            y_test,
            (probabilities >= 0.5).astype(int)
        )

        brier = brier_score_loss(
            y_test,
            probabilities
        )

        results_28.append({
            "test_year": test_year,
            "training_start": start_year,
            "train_examples": len(train),
            "test_examples": len(test),
            "accuracy": accuracy,
            "brier": brier
        })

        print(
            "Test year:", test_year,
            "| Training from:", start_year,
            "| Accuracy:", round(accuracy * 100, 2), "%",
            "| Brier:", round(brier, 4)
        )

# Summarise results across the four years
results_df_28 = pd.DataFrame(results_28)

summary_28 = results_df_28.groupby(
    "training_start"
).agg(
    average_accuracy=("accuracy", "mean"),
    average_brier=("brier", "mean")
)

print("\nAVERAGE RESULTS:")
print(summary_28.round(4).to_string())

# Confirm that all 12 model evaluations were completed
assert len(results_df_28) == 12

# Confirm the original model's 2022 reference score
reference_2022 = results_df_28[
    (results_df_28["test_year"] == 2022) &
    (results_df_28["training_start"] == 2015)
].iloc[0]

assert round(reference_2022["brier"], 4) == 0.1049

print("Test 28 passed!")


# Test 29: Compare baseline training periods

print("\nTEST 29: Historical baseline training periods")

baseline_results_29 = []

for test_year in [2020, 2021, 2022, 2023]:

    original_train, test = split_data_by_year(
        ml_data, test_year
    )

    for start_year in [2015, 2010, 2000]:

        if start_year == 2015:
            train = original_train.copy()

        else:
            older_train = ml_extended[
                (ml_extended["week_start"] >= pd.Timestamp(
                    f"{start_year}-01-01"
                )) &
                (ml_extended["week_start"] < pd.Timestamp(
                    "2015-01-01"
                ))
            ].copy()

            train = pd.concat(
                [original_train, older_train],
                ignore_index=True
            )

        # Calculate historical probability
        probabilities = rolling_baseline(train, test)

        # Evaluate with Brier Score
        brier = brier_score_loss(
            test["good_week"].astype(int),
            probabilities
        )

        baseline_results_29.append({
            "test_year": test_year,
            "training_start": start_year,
            "brier": brier
        })

        print(
            "Year:", test_year,
            "| Training from:", start_year,
            "| Brier:", round(brier, 4)
        )

results_29 = pd.DataFrame(baseline_results_29)

print("\nAVERAGE BASELINE RESULTS:")
print(
    results_29.groupby("training_start")["brier"]
    .mean()
    .round(4)
    .to_string()
)

assert len(results_29) == 12

print("Test 29 passed!")


# Test 30: Check availability of recent evaluation years

print("\nTEST 30: Check 2024-2025 data availability")

import sqlite3

connection = sqlite3.connect("snowproof.db")

recent_data = pd.read_sql_query(
    """
    SELECT station_code, date, snow_depth_cm
    FROM snow_depth_table
    WHERE date >= '2024-01-01'
      AND date < '2026-01-01'
    """,
    connection
)

connection.close()

recent_data["date"] = pd.to_datetime(
    recent_data["date"]
)

# Only the winter months used by Snow-Proof
recent_data = recent_data[
    recent_data["date"].dt.month.isin([12, 1, 2, 3, 4])
].copy()

# Only stations included in our ML model
recent_data = recent_data[
    recent_data["station_code"].isin(
        ml_data["station_code"].unique()
    )
].copy()

recent_data["year"] = recent_data["date"].dt.year

availability = recent_data.groupby("year").agg(
    total_days=("date", "size"),
    valid_days=("snow_depth_cm", "count"),
    stations=("station_code", "nunique")
)

print(availability.to_string())

assert set(availability.index) == {2024, 2025}

print("Test 30 passed!")


# Test 31: Count complete seven-day windows in 2024-2025

print("\nTEST 31: Complete ski weeks in 2024-2025")

recent_data = recent_data.sort_values(
    ["station_code", "date"]
).copy()

recent_groups = recent_data.groupby("station_code")

# Date six rows before each measurement
recent_data["week_start"] = recent_groups["date"].shift(6)

# Number of valid measurements in the last seven rows
recent_data["valid_count"] = recent_groups[
    "snow_depth_cm"
].transform(
    lambda values: values.notna().rolling(7).sum()
)

# Seven consecutive calendar days
recent_data["consecutive"] = (
    recent_data["date"] - recent_data["week_start"]
    == pd.Timedelta(days=6)
)

# Keep only complete seven-day windows
complete_weeks = recent_data[
    recent_data["consecutive"] &
    (recent_data["valid_count"] == 7)
].copy()

# Group by the year when the week starts
complete_weeks["year"] = complete_weeks["week_start"].dt.year

week_counts = complete_weeks.groupby("year").agg(
    complete_weeks=("week_start", "size"),
    stations=("station_code", "nunique")
)

print(week_counts.to_string())

print("\nTotal complete weeks:", len(complete_weeks))

assert len(complete_weeks) > 0

print("Test 31 passed!")


# Test 32: Prepare final evaluation labels for 2024-2025

print("\nTEST 32: Prepare recent evaluation labels")

# Calculate the seven-day average snow depth
recent_data["average_depth"] = recent_data.groupby(
    "station_code"
)["snow_depth_cm"].transform(
    lambda values: values.rolling(7).mean()
)

# Use only the complete weeks verified in Test 31
recent_labels = recent_data[
    recent_data["consecutive"] &
    (recent_data["valid_count"] == 7)
].copy()

# Apply our existing good-week definition
recent_labels["good_week"] = (
    recent_labels["average_depth"] >= 50
).astype(int)

# Add the same input features as our existing ML data
recent_labels["year"] = recent_labels["week_start"].dt.year
recent_labels["month"] = recent_labels["week_start"].dt.month
recent_labels["day"] = recent_labels["week_start"].dt.day

station_altitudes = ml_data[
    ["station_code", "altitude_m"]
].drop_duplicates("station_code")

recent_labels = recent_labels.merge(
    station_altitudes,
    on="station_code",
    how="left",
    validate="many_to_one"
)

recent_labels = recent_labels[
    [
        "station_code",
        "week_start",
        "year",
        "good_week",
        "altitude_m",
        "month",
        "day"
    ]
].sort_values(
    ["station_code", "week_start"]
).reset_index(drop=True)

# Verify the new evaluation dataset
assert len(recent_labels) == 5056
assert recent_labels["station_code"].nunique() == 18
assert recent_labels["altitude_m"].notna().all()
assert recent_labels["good_week"].isin([0, 1]).all()
assert not recent_labels.duplicated(
    ["station_code", "week_start"]
).any()

# Keep the final evaluation data separate
recent_labels.to_csv(
    "ml_test_2024_2025.csv",
    index=False
)

print("Evaluation examples:", len(recent_labels))
print("Stations:", recent_labels["station_code"].nunique())
print("Missing values:", recent_labels.isna().sum().sum())
print("Test 32 passed!")


# Test 33: Final evaluation on 2024-2025

print("\nTEST 33: Final evaluation 2024-2025")

from sklearn.base import clone

# Load the held-out evaluation data
final_test = pd.read_csv(
    "ml_test_2024_2025.csv",
    parse_dates=["week_start"]
)

# Training data must end before 2024
rf_train = ml_data.copy()
baseline_train = ml_extended.copy()

cutoff = pd.Timestamp("2024-01-01")

assert (
    rf_train["week_start"] + pd.Timedelta(days=6)
    < cutoff
).all()

assert (
    baseline_train["week_start"] + pd.Timedelta(days=6)
    < cutoff
).all()

assert (final_test["week_start"] >= cutoff).all()

# All test stations must be known from training
assert set(final_test["station_code"]).issubset(
    set(rf_train["station_code"])
)

assert set(final_test["station_code"]).issubset(
    set(baseline_train["station_code"])
)

# Train the original Random Forest on 2015-2023
final_rf = clone(model)

final_rf.fit(
    rf_train[features],
    rf_train["good_week"].astype(int)
)

# Predict all held-out weeks with Random Forest
final_test["rf_probability"] = final_rf.predict_proba(
    final_test[features]
)[:, 1]

# Predict the same weeks using the historical baseline
final_test["baseline_probability"] = rolling_baseline(
    baseline_train,
    final_test
)

# Check that all predictions are valid probabilities
assert final_test[
    ["rf_probability", "baseline_probability"]
].notna().all().all()

for column in ["rf_probability", "baseline_probability"]:
    assert final_test[column].between(0, 1).all()

# Compare the methods separately for each year
for year in [2024, 2025]:

    test_year = final_test[
        final_test["year"] == year
    ]

    actual = test_year["good_week"].astype(int)

    print("\nYear:", year)
    print("Test examples:", len(test_year))

    for method in ["rf_probability", "baseline_probability"]:

        probabilities = test_year[method]

        accuracy = accuracy_score(
            actual,
            (probabilities >= 0.5).astype(int)
        )

        brier = brier_score_loss(
            actual,
            probabilities
        )

        print(
            method,
            "| Accuracy:", round(accuracy * 100, 2), "%",
            "| Brier:", round(brier, 4)
        )

# Overall evaluation across both years
print("\nOVERALL RESULTS 2024-2025")

actual = final_test["good_week"].astype(int)

for method in ["rf_probability", "baseline_probability"]:

    probabilities = final_test[method]

    accuracy = accuracy_score(
        actual,
        (probabilities >= 0.5).astype(int)
    )

    brier = brier_score_loss(
        actual,
        probabilities
    )

    print(
        method,
        "| Accuracy:", round(accuracy * 100, 2), "%",
        "| Brier:", round(brier, 4)
    )

assert len(final_test) == 5056

print("Test 33 passed!")


# Test 34: Inspect prediction quality and class balance

print("\nTEST 34: Prediction distribution 2024-2025")

from sklearn.metrics import confusion_matrix

for year in [2024, 2025]:

    test_year = final_test[
        final_test["year"] == year
    ].copy()

    actual = test_year["good_week"].astype(int)

    predicted = (
        test_year["rf_probability"] >= 0.5
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    ).ravel()

    print("\nYear:", year)
    print("Actual good weeks:", int(actual.sum()))
    print("Actual bad weeks:", int((actual == 0).sum()))
    print("Share of good weeks:", round(actual.mean() * 100, 2), "%")
    print("Predicted good weeks:", int(predicted.sum()))
    print("Average RF probability:", round(
        test_year["rf_probability"].mean() * 100, 2
    ), "%")

    print("Correctly predicted good:", tp)
    print("Correctly predicted bad:", tn)
    print("Incorrectly predicted good:", fp)
    print("Incorrectly predicted bad:", fn)

print("\nTest 34 passed!")


# Test 35: Check Random Forest probability calibration

print("\nTEST 35: Random Forest calibration")

# Use the existing predictions from Test 33
calibration_data = final_test.copy()

# Divide predictions into five probability groups
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

labels = [
    "0-20%",
    "20-40%",
    "40-60%",
    "60-80%",
    "80-100%"
]

calibration_data["probability_group"] = pd.cut(
    calibration_data["rf_probability"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# Compare predicted probabilities with actual outcomes
calibration = calibration_data.groupby(
    "probability_group",
    observed=True
).agg(
    weeks=("good_week", "size"),
    predicted_probability=("rf_probability", "mean"),
    actual_good_rate=("good_week", "mean")
)

# Convert values to percentages
calibration["predicted_percent"] = (
    calibration["predicted_probability"] * 100
).round(2)

calibration["actual_percent"] = (
    calibration["actual_good_rate"] * 100
).round(2)

# Positive difference means the model was too optimistic
calibration["difference"] = (
    calibration["predicted_percent"]
    - calibration["actual_percent"]
).round(2)

# Display results
print(
    calibration[
        [
            "weeks",
            "predicted_percent",
            "actual_percent",
            "difference"
        ]
    ].to_string()
)

# Verify that all test examples were included
assert calibration["weeks"].sum() == 5056

print("\nTest 35 passed!")


# Test 36: Calibration on development years 2020-2023

print("\nTEST 36: Development-year calibration")

development_predictions = []

for year in [2020, 2021, 2022, 2023]:

    train, test = split_data_by_year(ml_data, year)

    # Original model settings and original training order
    development_model = clone(model)

    development_model.fit(
        train[features],
        train["good_week"].astype(int)
    )

    predictions = test[
        ["station_code", "week_start", "good_week"]
    ].copy()

    predictions["rf_probability"] = (
        development_model.predict_proba(
            test[features]
        )[:, 1]
    )

    development_predictions.append(predictions)

development = pd.concat(
    development_predictions,
    ignore_index=True
)

# Use the same probability groups as Test 35
development["probability_group"] = pd.cut(
    development["rf_probability"],
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    include_lowest=True
)

dev_calibration = development.groupby(
    "probability_group",
    observed=True
).agg(
    weeks=("good_week", "size"),
    predicted=("rf_probability", "mean"),
    actual=("good_week", "mean")
)

dev_calibration["predicted"] *= 100
dev_calibration["actual"] *= 100

print(dev_calibration.round(2).to_string())

assert dev_calibration["weeks"].sum() == len(development)

print("\nTotal development examples:", len(development))
print("Test 36 passed!")


# Test 37: Check historical_average function

print("\nTEST 37: Historical average +/-14 days")

from streamlit_app import historical_average

example_labels = pd.DataFrame({
    "station_code": ["DAV", "DAV", "DAV", "DAV", "ZER"],
    "week_start": pd.to_datetime([
        "2020-02-01",
        "2021-02-10",
        "2022-02-20",
        "2023-04-01",
        "2021-02-10"
    ]),
    "good_week": [1, 1, 0, 0, 0]
})

result = historical_average(
    example_labels,
    "DAV",
    2,
    10,
    2027
)

print("Historical probability:", result)
print("Expected probability:", 2 / 3)

assert abs(result - 2 / 3) < 0.000001

print("Test 37 passed!")


# Test 38: Compare app evaluation with experiment results

print("\nTEST 38: Verify app test_model results")

from streamlit_app import test_model

# Prepare the same labels and station information
# that our experiments use
labels_38 = ml_data[
    ["station_code", "week_start", "good_week"]
].copy()

stations_38 = ml_data[
    ["station_code", "altitude_m"]
].drop_duplicates("station_code").copy()

# Run the actual evaluation function from our app
results_38 = test_model(labels_38, stations_38)

# Reference results from our earlier experiments
reference_38 = {
    2020: (0.1141, 0.1103),
    2021: (0.1739, 0.1510),
    2022: (0.1049, 0.0784),
    2023: (0.1510, 0.1921)
}

print("\nAPP EVALUATION RESULTS:")
print(results_38.to_string(index=False))

print("\nCOMPARISON WITH REFERENCE:")

assert len(results_38) == 4

for _, row in results_38.iterrows():

    year = int(row["test_year"])

    expected_rf, expected_baseline = reference_38[year]

    rf_difference = abs(
        row["rf_brier"] - expected_rf
    )

    baseline_difference = abs(
        row["baseline_brier"] - expected_baseline
    )

    print(
        "Year:", year,
        "| RF matches:", rf_difference < 0.0002,
        "| Baseline matches:", baseline_difference < 0.0002
    )

    assert rf_difference < 0.0002
    assert baseline_difference < 0.0002

print("\nTest 38 passed!")



# Test 39: Historical average with real extended labels

print("\nTEST 39: Historical average with real data")

from streamlit_app import historical_average

historical_labels = pd.read_csv(
    "ml_dataset_extended.csv",
    parse_dates=["week_start"]
)

# Example: Davos, February 10, 2027
probability = historical_average(
    historical_labels,
    "DAV",
    2,
    10,
    2027
)

print("Station: DAV")
print("Requested date: 2027-02-10")
print("Historical probability:", probability)

assert probability is not None
assert 0 <= probability <= 1

print("Test 39 passed!")
