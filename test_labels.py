import pandas as pd
from data import load_snow_depth, load_stations
from streamlit_app import train_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import brier_score_loss
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data import load_resorts, link_stations_to_resorts

def check_good_week(snow_depth):

    # Check that exactly 7 days are available
    if len(snow_depth) != 7:
        return None

    # Check whether any snow measurement is missing
    if pd.isna(snow_depth).any():
        return None

    # Calculate average snow depth
    average = sum(snow_depth) / 7

    # Decide whether it was a good week
    if average >= 50:
        return 1
    else:
        return 0


def label_good_weeks(snow_data):

    # Sort measurements by station and date
    snow_data = snow_data.sort_values(["station_code", "date"])

    # Store all evaluated weeks
    results = []

    # Go through every station separately
    for station, station_data in snow_data.groupby("station_code"):

        station_data = station_data.reset_index(drop=True)

        # Check every possible seven-day period
        for i in range(len(station_data) - 6):

            week_data = station_data.iloc[i:i+7]

            start_date = week_data["date"].iloc[0]
            end_date = week_data["date"].iloc[-1]

            
            # Check that all seven dates are consecutive
            date_differences = week_data["date"].diff().dropna()

            if not (date_differences == pd.Timedelta(days=1)).all():
                continue

                      

            # Only evaluate weeks fully within December-April
            if not all(month in [12, 1, 2, 3, 4]
                       for month in week_data["date"].dt.month):
                continue

            # Calculate the label
            week = week_data["snow_depth_cm"].tolist()
            label = check_good_week(week)

            # Save the result
            results.append({
                "station_code": station,
                "week_start": start_date,
                "year": start_date.year,
                "good_week": label
            })

    return pd.DataFrame(results)



# Test 1: Good ski week
week_1 = [70, 67, 61, 59, 77, 75, 71]

# Test 2: Bad ski week
week_2 = [20, 30, 25, 40, 35, 30, 25]

# Test 3: Incomplete ski week
week_3 = [70, 65, 60, 55, 50]

print("Week 1:", check_good_week(week_1))
print("Week 2:", check_good_week(week_2))
print("Week 3:", check_good_week(week_3))

# Test 4: Seven days, but one measurement is missing
week_4 = [70, 65, 60, None, 55, 50, 65]

print("Week 4:", check_good_week(week_4))

# Test 5: Real snow measurements from MeteoSwiss
snow_data = load_snow_depth("DAV", "2020-02-01", "2020-02-07")

# Extract snow depth values
snow_values = snow_data["snow_depth_cm"].tolist()

# Check whether it was a good ski week
result = check_good_week(snow_values)

print("Real Davos data:", snow_values)
print("Good ski week:", result)



# Test 6: Create a table of good ski weeks

# Load snow measurements for February 2020
snow_data = load_snow_depth("DAV", "2020-02-01", "2020-02-29")

# Create an empty list to store the results
results = []

# Go through all possible seven-day periods
for i in range(len(snow_data) - 6):

    # Select seven consecutive measurements
    week = snow_data["snow_depth_cm"].iloc[i:i+7].tolist()

    # Calculate whether the week was good
    label = check_good_week(week)

    # Get the start date
    start_date = snow_data["date"].iloc[i]

    # Save the result
    results.append({
        "station_code": "DAV",
        "week_start": start_date,
        "year": start_date.year,
        "good_week": label
    })

# Convert the list into a Pandas DataFrame
labels = pd.DataFrame(results)

# Show the table
print(labels)


# Test 7: Check data quality

# How many rows do we have?
print("Number of rows:", len(snow_data))

# Are any snow measurements missing?
print("Missing measurements:", snow_data["snow_depth_cm"].isna().sum())

# Are any dates duplicated?
print("Duplicate dates:", snow_data["date"].duplicated().sum())

# Check whether dates are consecutive
date_differences = snow_data["date"].diff()

print("Date differences:")
print(date_differences.value_counts())

# Test 8: Use the new function with real Davos data
labels_test = label_good_weeks(snow_data)

print("Labels created by the new function:")
print(labels_test)


# Test 9: Check whether missing dates are detected

# Create a copy of the original data
incomplete_data = snow_data.copy()

# Remove 10 February 2020
incomplete_data = incomplete_data[
    incomplete_data["date"] != "2020-02-10"
]

# Create labels using the incomplete data
incomplete_labels = label_good_weeks(incomplete_data)

print("Original number of labels:", len(labels_test))
print("Number of labels after removing one day:", len(incomplete_labels))
print(incomplete_labels)



# Task 6: Historical average


def historical_average(labels, station_code, month, day, prediction_year):

    # Select the requested station
    station_labels = labels[
        labels["station_code"] == station_code
    ]

    # Select the same month and day in previous years
    same_date = station_labels[
        (station_labels["week_start"].dt.month == month) &
        (station_labels["week_start"].dt.day == day) &
        (station_labels["week_start"].dt.year < prediction_year)
    ]

    # Remove unknown weeks
    known_weeks = same_date[
        same_date["good_week"].notna()
    ]

    # No historical observations available
    if len(known_weeks) == 0:
        return None

    # Calculate the historical share of good weeks
    return known_weeks["good_week"].mean()



# Test 10: Historical average with example data

example_labels = pd.DataFrame({
    "station_code": ["DAV", "DAV", "DAV", "DAV", "DAV"],
    "week_start": pd.to_datetime([
        "2019-02-10",
        "2020-02-10",
        "2021-02-10",
        "2022-02-10",
        "2023-02-10"
    ]),
    "good_week": [1, 1, 0, 1, 0]
})

result = historical_average(example_labels, "DAV", 2, 10, 2024)

print("Historical average:", result)

# Test 11: Only use data from before 2022
result_2022 = historical_average(example_labels, "DAV", 2, 10, 2022)

print("Historical average before 2022:", result_2022)


# Test 12: Historical average with real Davos data

# Load snow measurements for several years
historical_snow = load_snow_depth(
    "DAV", "2015-01-01", "2024-12-31"
)

# Turn daily measurements into good-week labels
historical_labels = label_good_weeks(historical_snow)

# Calculate the historical average for 10 February
result = historical_average(
    historical_labels, "DAV", 2, 10, 2025
)

print("Number of historical labels:", len(historical_labels))
print("Historical average for Davos:", result)


# Test 13: Check which years were used

# Select Davos weeks starting on 10 February
february_weeks = historical_labels[
    (historical_labels["station_code"] == "DAV") &
    (historical_labels["week_start"].dt.month == 2) &
    (historical_labels["week_start"].dt.day == 10) &
    (historical_labels["week_start"].dt.year < 2025)
]

# Show the years and their labels
print(february_weeks[["year", "good_week"]])

# Count valid observations
print("Number of valid years:",
      february_weeks["good_week"].count())


# Test 14: Check the distribution of good and bad ski weeks

# Keep only weeks with a known result
valid_labels = historical_labels[
    historical_labels["good_week"].notna()
]

# Count good and bad weeks
print("Good and bad ski weeks:")
print(valid_labels["good_week"].value_counts())

# Calculate their proportions
print("Proportions:")
print(valid_labels["good_week"].value_counts(normalize=True))

# Count the years covered by the data
print("Number of years:")
print(valid_labels["year"].nunique())


# Test 15: Explore available snow measurement stations

# Load all available stations
stations = load_stations()

# Show the first five stations
print("First five stations:")
print(stations.head())

# Show the column names
print("Available columns:")
print(stations.columns)

# Count the stations
print("Total number of stations:", len(stations))


# Test 16: Explore station altitudes

# Show statistics about station altitudes
print("Altitude statistics:")
print(stations["altitude_m"].describe())

# Count stations in different altitude groups
low_stations = stations[stations["altitude_m"] < 1000]

middle_stations = stations[
    (stations["altitude_m"] >= 1000) &
    (stations["altitude_m"] < 2000)
]

high_stations = stations[stations["altitude_m"] >= 2000]

print("Stations below 1000 m:", len(low_stations))
print("Stations between 1000 and 2000 m:", len(middle_stations))
print("Stations above 2000 m:", len(high_stations))


# Test 17: Find stations at higher altitudes

# Select stations at 1500 metres or higher
mountain_stations = stations[
    stations["altitude_m"] >= 1500
]

# Sort by altitude, highest first
mountain_stations = mountain_stations.sort_values(
    "altitude_m", ascending=False
)

# Show their names, codes and altitudes
print("Mountain stations:")
print(mountain_stations[
    ["station_code", "station_name", "altitude_m"]
].to_string(index=False))

print("Number of mountain stations:", len(mountain_stations))


# Test 18: Check historical data from two more stations

station_codes = ["ABO", "ZER", "SAF"]

for station_code in station_codes:

    print("Station:", station_code)

    # Load historical snow measurements
    snow = load_snow_depth(
        station_code, "2015-01-01", "2024-12-31"
    )

    # Check the measurements
    print("Number of measurements:", len(snow))
    print("Missing snow measurements:",
          snow["snow_depth_cm"].isna().sum())

    # Create good-week labels
    labels = label_good_weeks(snow)

    # Count good and bad weeks
    print("Good and bad weeks:")
    print(labels["good_week"].value_counts())

    print("------------------------")



# Test 20: Check snow depth values

station_codes = ["ABO", "DAV", "ZER", "ARO", "WFJ", "SAF"]

for station_code in station_codes:
    snow = load_snow_depth(
        station_code, "2015-01-01", "2024-12-31"
    )

    print("Snow depth statistics for:", station_code)
    print(snow["snow_depth_cm"].describe())
    print("------------------------")



# Test 21: Average snow depth during ski season

station_codes = ["ABO", "DAV", "ZER", "ARO", "WFJ"]

for station_code in station_codes:

    snow = load_snow_depth(
        station_code, "2015-01-01", "2024-12-31"
    )

    # Keep only December to April
    winter_snow = snow[
        snow["date"].dt.month.isin([12, 1, 2, 3, 4])
    ]

    print("Station:", station_code)
    print("Winter average snow depth:",
          winter_snow["snow_depth_cm"].mean())
    print("Winter maximum snow depth:",
          winter_snow["snow_depth_cm"].max())
    print("------------------------")


# Test 22: Build a first machine learning dataset

station_codes = ["ABO", "DAV", "ZER", "ARO", "WFJ"]

all_labels = []

for station_code in station_codes:

    # Load historical snow measurements
    snow = load_snow_depth(
        station_code, "2015-01-01", "2024-12-31"
    )

    # Create good-week labels
    labels = label_good_weeks(snow)

    # Add station altitude
    station_info = stations[
        stations["station_code"] == station_code
    ]

    altitude = station_info["altitude_m"].iloc[0]

    labels["altitude_m"] = altitude

    # Add starting month and day
    labels["month"] = labels["week_start"].dt.month
    labels["day"] = labels["week_start"].dt.day

    all_labels.append(labels)

# Combine all stations into one DataFrame
ml_data = pd.concat(all_labels, ignore_index=True)

# Remove weeks with missing labels
ml_data = ml_data.dropna(subset=["good_week"])

print("First rows of the ML dataset:")
print(ml_data.head())

print("Total number of ML examples:", len(ml_data))

print("Good and bad weeks:")
print(ml_data["good_week"].value_counts())


# Test 23: Split the data into training, validation and test sets

# Calculate the last day of each seven-day period
ml_data["week_end"] = ml_data["week_start"] + pd.Timedelta(days=6)

# Training: weeks entirely within 2015–2021
train_data = ml_data[
    ml_data["week_end"] < "2022-01-01"
]

# Validation: weeks entirely within 2022–2023
validation_data = ml_data[
    (ml_data["week_start"] >= "2022-01-01") &
    (ml_data["week_end"] < "2024-01-01")
]

# Test: weeks starting in 2024
test_data = ml_data[
    ml_data["week_start"] >= "2024-01-01"
]

print("Training examples:", len(train_data))
print("Validation examples:", len(validation_data))
print("Test examples:", len(test_data))

print("Training labels:")
print(train_data["good_week"].value_counts())

print("Validation labels:")
print(validation_data["good_week"].value_counts())

print("Test labels:")
print(test_data["good_week"].value_counts())


# Test 24: Train our first classification model

# Choose the information our model can use
features = ["altitude_m", "month", "day"]

# Separate inputs (X) and correct answers (y)
X_train = train_data[features]
y_train = train_data["good_week"].astype(int)

X_validation = validation_data[features]
y_validation = validation_data["good_week"].astype(int)

# Scale the inputs using training data only
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)

# Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Make predictions for the validation years
predictions = model.predict(X_validation_scaled)

# Evaluate the predictions
accuracy = accuracy_score(y_validation, predictions)

print("Validation accuracy:", accuracy)
print("Confusion matrix:")
print(confusion_matrix(y_validation, predictions))


# Test 25: Evaluate predicted probabilities

# Predict probabilities for the validation data
probabilities = model.predict_proba(X_validation_scaled)[:, 1]

# Calculate the Brier Score
brier = brier_score_loss(y_validation, probabilities)

print("Brier Score:", brier)

# Show some example predictions
print("First 10 predicted probabilities:")
print(probabilities[:10])

print("First 10 actual results:")
print(y_validation.iloc[:10].tolist())


# Test 26: Compare ML with a simple baseline

# How often were training weeks good?
baseline_probability = y_train.mean()

# Use this same probability for all validation weeks
baseline_probabilities = [baseline_probability] * len(y_validation)

# Calculate the baseline Brier Score
baseline_brier = brier_score_loss(
    y_validation, baseline_probabilities
)

print("Baseline probability:", baseline_probability)
print("Baseline Brier Score:", baseline_brier)
print("ML Brier Score:", brier)


# Test 27: Compare ML with historical probabilities

historical_probabilities = []

for _, week in validation_data.iterrows():

    # Find matching weeks in the training data
    same_station = train_data[
        train_data["station_code"] == week["station_code"]
    ]

    same_date = same_station[
        (same_station["month"] == week["month"]) &
        (same_station["day"] == week["day"])
    ]

    # Calculate historical probability
    if len(same_date) > 0:
        probability = same_date["good_week"].mean()
    else:
        probability = y_train.mean()

    historical_probabilities.append(probability)

# Evaluate the historical method
historical_brier = brier_score_loss(
    y_validation, historical_probabilities
)

print("Historical Brier Score:", historical_brier)
print("ML Brier Score:", brier)


# Test 28: Represent the year as a circle

# Calculate the day number within the year
ml_data["day_of_year"] = ml_data["week_start"].dt.dayofyear

# Turn the day number into a position on a circle
ml_data["season_sin"] = np.sin(
    2 * np.pi * ml_data["day_of_year"] / 365.25
)

ml_data["season_cos"] = np.cos(
    2 * np.pi * ml_data["day_of_year"] / 365.25
)

print("New seasonal features:")
print(ml_data[
    ["week_start", "day_of_year", "season_sin", "season_cos"]
].head())


# Test 29: Train logistic regression with seasonal features

# Update training and validation data with the new columns
train_data = ml_data[
    ml_data["week_end"] < "2022-01-01"
].copy()

validation_data = ml_data[
    (ml_data["week_start"] >= "2022-01-01") &
    (ml_data["week_end"] < "2024-01-01")
].copy()

# Choose the new features
new_features = ["altitude_m", "season_sin", "season_cos"]

X_train_new = train_data[new_features]
X_validation_new = validation_data[new_features]

# Scale the new features
new_scaler = StandardScaler()

X_train_new_scaled = new_scaler.fit_transform(X_train_new)
X_validation_new_scaled = new_scaler.transform(X_validation_new)

# Train a new model
new_model = LogisticRegression(max_iter=1000)
new_model.fit(X_train_new_scaled, y_train)

# Predict probabilities
new_probabilities = new_model.predict_proba(
    X_validation_new_scaled
)[:, 1]

# Calculate the new Brier Score
new_brier = brier_score_loss(
    y_validation, new_probabilities
)

print("Old ML Brier Score:", brier)
print("New ML Brier Score:", new_brier)
print("Historical Brier Score:", historical_brier)


# Test 30: Train a Random Forest model

# Use the original three features
features = ["altitude_m", "month", "day"]

X_train_rf = train_data[features]
X_validation_rf = validation_data[features]

# Create the Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=20,
    random_state=42
)

# Train the model
rf_model.fit(X_train_rf, y_train)

# Predict probabilities
rf_probabilities = rf_model.predict_proba(
    X_validation_rf
)[:, 1]

# Calculate Brier Score
rf_brier = brier_score_loss(
    y_validation, rf_probabilities
)

# Calculate accuracy
rf_predictions = rf_model.predict(X_validation_rf)
rf_accuracy = accuracy_score(y_validation, rf_predictions)

print("Random Forest Brier Score:", rf_brier)
print("Random Forest Accuracy:", rf_accuracy)
print("Old Logistic Regression Brier Score:", brier)
print("Historical Brier Score:", historical_brier)


# Test 31: Evaluate Random Forest classification errors

from sklearn.metrics import precision_score, recall_score, f1_score

print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_validation, rf_predictions))

print("Random Forest Precision:",
      precision_score(y_validation, rf_predictions))

print("Random Forest Recall:",
      recall_score(y_validation, rf_predictions))

print("Random Forest F1 Score:",
      f1_score(y_validation, rf_predictions))


# Test 32: Check Random Forest accuracy for each station

# Copy validation data and add model predictions
station_results = validation_data.copy()
station_results["prediction"] = rf_predictions

# Check each station separately
station_codes = ["ABO", "DAV", "ZER", "ARO", "WFJ"]

for station_code in station_codes:

    # Select only this station
    station = station_results[
        station_results["station_code"] == station_code
    ]

    # Compare predicted and actual results
    accuracy = accuracy_score(
        station["good_week"],
        station["prediction"]
    )

    print("Station:", station_code)
    print("Accuracy:", round(accuracy * 100, 1), "%")
    print("------------------------")


# Test 33: Compare actual and predicted good weeks by station

for station_code in station_codes:

    station = station_results[
        station_results["station_code"] == station_code
    ]

    print("Station:", station_code)

    # Actual good and bad weeks
    print("Actual results:")
    print(station["good_week"].value_counts())

    # Predicted good and bad weeks
    print("Model predictions:")
    print(station["prediction"].value_counts())

    print("------------------------")


# Test 34: Compare Random Forest with a simple station rule

baseline_predictions = []

for _, week in validation_data.iterrows():

    # Look at historical training weeks from the same station
    station_history = train_data[
        train_data["station_code"] == week["station_code"]
    ]

    # Was the majority of these weeks good?
    average = station_history["good_week"].mean()

    if average >= 0.5:
        baseline_predictions.append(1)
    else:
        baseline_predictions.append(0)

# Compare both methods
baseline_accuracy = accuracy_score(
    y_validation, baseline_predictions
)

print("Station baseline accuracy:",
      round(baseline_accuracy * 100, 1), "%")

print("Random Forest accuracy:",
      round(rf_accuracy * 100, 1), "%")


# Test 35: Train the model using the Streamlit function

print("TEST 35: Training our model from streamlit_app.py")

# Select the columns required by train_model()
app_labels = ml_data[
    ["station_code", "week_start", "year", "good_week"]
].copy()

# Train the model using our new function
app_model = train_model(app_labels, stations)

# Check whether training worked
print("Model type:", type(app_model).__name__)
print("Features:", app_model.feature_names_in_)
print("Model trained successfully!")


# Test 36: Check stored station information

print("\nTEST 36: Stored station information")

# Show which stations were used for training
print("Trained stations:")
print(app_model.snowproof_station_codes)

# Show the altitude of each trained station
print("Station altitudes:")
print(app_model.snowproof_station_altitudes)


# Test 37: Connect ski resorts to snow stations

print("\nTEST 37: Resort and station connection")

# Load ski resorts
resorts = load_resorts()

# Connect resorts with nearby snow stations
linked_resorts = link_stations_to_resorts(resorts, stations)

# Show available columns and first five resorts
print("Columns:", linked_resorts.columns.tolist())
print(linked_resorts.head())
