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



# Test 38: Check which resorts use trained stations

print("\nTEST 38: Supported ski resorts")

# Keep resorts whose stations were used for training
supported_resorts = linked_resorts[
    linked_resorts["station_code"].isin(
        app_model.snowproof_station_codes
    )
]

# Show their names and stations
print(supported_resorts[["name", "station_code"]])

# Count how many resorts are supported
print("Supported resorts:", len(supported_resorts))
print("Total resorts:", len(linked_resorts))


# Test 39: Show all resort-station connections

print("\nTEST 39: All resort-station connections")

print(
    linked_resorts[
        ["resort_id", "name", "station_code"]
    ].to_string(index=False)
)

# Test 40: Check historical snow data for Laax station

print("\nTEST 40: Snow data quality for FLI")

# Load historical snow measurements
fli_snow = load_snow_depth(
    "FLI", "2015-01-01", "2024-12-31"
)

# Count all measurements
print("Total measurements:", len(fli_snow))

# Count missing snow depth values
print(
    "Missing measurements:",
    fli_snow["snow_depth_cm"].isna().sum()
)

# Count available snow depth values
print(
    "Valid measurements:",
    fli_snow["snow_depth_cm"].notna().sum()
)


# Test 41: Check valid ski weeks for FLI

print("\nTEST 41: Valid ski weeks for FLI")

# Create seven-day ski week labels
fli_labels = label_good_weeks(fli_snow)

# Count weeks with known and unknown results
print("Total ski week windows:", len(fli_labels))

print(
    "Valid ski weeks:",
    fli_labels["good_week"].notna().sum()
)

print(
    "Unknown ski weeks:",
    fli_labels["good_week"].isna().sum()
)


# Test 42: Check valid ski weeks per year for FLI

print("\nTEST 42: Valid ski weeks by year")

# Keep only weeks with a known label
valid_fli_labels = fli_labels.dropna(subset=["good_week"])

# Count valid weeks for each year
print(
    valid_fli_labels.groupby("year").size()
)


# Test 43: Prepare additional stations for data quality checks

print("\nTEST 43: Additional station codes")

# Get all assigned stations and remove duplicates
additional_stations = linked_resorts["station_code"].dropna().unique()

# Remove stations already used in our model
additional_stations = [
    code for code in additional_stations
    if code not in app_model.snowproof_station_codes
]

print("Additional stations:", additional_stations)
print("Number of additional stations:", len(additional_stations))


# Test 44: Check snow data quality for all additional stations

print("\nTEST 44: Snow data quality of additional stations")

for code in additional_stations:

    # Download or load historical measurements
    snow = load_snow_depth(
        code, "2015-01-01", "2023-12-31"
    )

    # Count available measurements
    valid_days = snow["snow_depth_cm"].notna().sum()

    # Create ski week labels
    week_labels = label_good_weeks(snow)

    # Count usable ski weeks
    valid_weeks = week_labels["good_week"].notna().sum()

    # Show the results for this station
    print(
        code,
        "| Valid days:", valid_days,
        "| Valid weeks:", valid_weeks
    )


# Test 45: Prepare labels from additional stations

print("\nTEST 45: Prepare extended ML dataset")

# Exclude the station without usable snow measurements
usable_stations = [
    code for code in additional_stations
    if code != "SAF"
]

# Collect labels from the additional stations
new_labels_list = []

for code in usable_stations:

    # Load snow measurements
    snow = load_snow_depth(
        code, "2015-01-01", "2023-12-31"
    )

    # Create good-week labels
    labels = label_good_weeks(snow)

    # Keep only labels with a known result
    labels = labels.dropna(subset=["good_week"])

    # Add the labels to our list
    new_labels_list.append(labels)

# Combine all additional station labels
new_labels = pd.concat(new_labels_list, ignore_index=True)

print("Additional stations:", len(usable_stations))
print("Additional valid week labels:", len(new_labels))
print("First five rows:")
print(new_labels.head())


# Test 46: Combine old and new ML labels

print("\nTEST 46: Combine old and new labels")

# Keep old labels whose seven-day window ends before 2024
old_labels = ml_data[
    ml_data["week_start"] + pd.Timedelta(days=6)
    < pd.Timestamp("2024-01-01")
][["station_code", "week_start", "year", "good_week"]].copy()

# Combine the old and new station labels
extended_labels = pd.concat(
    [old_labels, new_labels],
    ignore_index=True
)

# Check our combined dataset
print("Old labels:", len(old_labels))
print("New labels:", len(new_labels))
print("Total labels:", len(extended_labels))
print("Stations:", extended_labels["station_code"].nunique())
print("Latest week start:", extended_labels["week_start"].max())


# Test 47: Prepare features for extended Random Forest

print("\nTEST 47: Prepare extended ML features")

# Add station altitudes
extended_ml_data = extended_labels.merge(
    stations[["station_code", "altitude_m"]],
    on="station_code",
    how="inner"
)

# Extract month and day from the date
extended_ml_data["month"] = extended_ml_data["week_start"].dt.month
extended_ml_data["day"] = extended_ml_data["week_start"].dt.day

# Check the results
print("Total ML examples:", len(extended_ml_data))
print("Number of stations:", extended_ml_data["station_code"].nunique())
print("Missing altitudes:", extended_ml_data["altitude_m"].isna().sum())

print("\nFirst five rows:")
print(extended_ml_data.head())


# Test 48: Split the extended dataset by year

print("\nTEST 48: Training and validation split")

# Training: weeks ending before 2022
extended_train = extended_ml_data[
    extended_ml_data["week_start"] + pd.Timedelta(days=6)
    < pd.Timestamp("2022-01-01")
].copy()

# Validation: weeks starting in 2022 or 2023
extended_val = extended_ml_data[
    extended_ml_data["week_start"] >= pd.Timestamp("2022-01-01")
].copy()

print("Training examples:", len(extended_train))
print("Validation examples:", len(extended_val))

print("Training stations:", extended_train["station_code"].nunique())
print("Validation stations:", extended_val["station_code"].nunique())

print(
    "Latest training week end:",
    (extended_train["week_start"] + pd.Timedelta(days=6)).max()
)

print(
    "Earliest validation week:",
    extended_val["week_start"].min()
)


# Test 49: Find stations missing in validation

print("\nTEST 49: Missing validation stations")

train_station_codes = set(extended_train["station_code"])
val_station_codes = set(extended_val["station_code"])

missing_stations = train_station_codes - val_station_codes

print("Stations without validation data:", missing_stations)

# Show number of available examples per year
for code in missing_stations:
    print("\nStation:", code)
    station_data = extended_ml_data[
        extended_ml_data["station_code"] == code
    ]
    print(station_data.groupby("year").size())


# Test 50: Train the extended Random Forest

from sklearn.ensemble import RandomForestClassifier

print("\nTEST 50: Train extended Random Forest")

# The three inputs our model learns from
features = ["altitude_m", "month", "day"]

# Training inputs and correct answers
X_extended_train = extended_train[features]
y_extended_train = extended_train["good_week"].astype(int)

# Create the Random Forest
extended_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=20,
    random_state=42
)

# Train the model
extended_rf.fit(X_extended_train, y_extended_train)

print("Extended Random Forest trained successfully!")
print("Training examples:", len(X_extended_train))
print("Training stations:", extended_train["station_code"].nunique())
print("Number of trees:", extended_rf.n_estimators)


# Test 51: Compare both Random Forest models

from sklearn.metrics import accuracy_score, brier_score_loss

print("\nTEST 51: Compare Random Forest models")

features = ["altitude_m", "month", "day"]

# Use exactly the same validation examples
original_codes = ["ABO", "DAV", "ZER", "ARO", "WFJ"]

common_val = extended_val[
    extended_val["station_code"].isin(original_codes)
].copy()

X_common = common_val[features]
y_common = common_val["good_week"].astype(int)

# Predictions from the original Random Forest
old_prob = rf_model.predict_proba(X_common)[:, 1]
old_pred = (old_prob >= 0.5).astype(int)

# Predictions from the extended Random Forest
new_prob = extended_rf.predict_proba(X_common)[:, 1]
new_pred = (new_prob >= 0.5).astype(int)

print("Validation examples:", len(common_val))

print("\nOriginal Random Forest (5 stations)")
print("Accuracy:", round(accuracy_score(y_common, old_pred), 4))
print("Brier Score:", round(brier_score_loss(y_common, old_prob), 4))

print("\nExtended Random Forest (19 stations)")
print("Accuracy:", round(accuracy_score(y_common, new_pred), 4))
print("Brier Score:", round(brier_score_loss(y_common, new_prob), 4))


# Test 52: Evaluate extended Random Forest on all stations

print("\nTEST 52: Extended model on all validation stations")

# Prepare validation inputs and actual results
X_all_val = extended_val[features]
y_all_val = extended_val["good_week"].astype(int)

# Predict probabilities and classes
all_prob = extended_rf.predict_proba(X_all_val)[:, 1]
all_pred = (all_prob >= 0.5).astype(int)

# Evaluate results
print("Validation examples:", len(extended_val))
print("Validation stations:", extended_val["station_code"].nunique())

print("Accuracy:", round(accuracy_score(y_all_val, all_pred), 4))
print("Brier Score:", round(brier_score_loss(y_all_val, all_prob), 4))


# Test 53: Evaluate extended Random Forest per station

print("\nTEST 53: Accuracy and Brier Score per station")

# Create a copy of our validation data
station_evaluation = extended_val.copy()

# Add predictions and probabilities
station_evaluation["prediction"] = all_pred
station_evaluation["probability"] = all_prob

# Evaluate every station separately
for code, data in station_evaluation.groupby("station_code"):

    accuracy = accuracy_score(
        data["good_week"].astype(int),
        data["prediction"]
    )

    brier = brier_score_loss(
        data["good_week"].astype(int),
        data["probability"]
    )

    print(
        code,
        "| Examples:", len(data),
        "| Accuracy:", round(accuracy * 100, 1), "%",
        "| Brier:", round(brier, 4)
    )


# Test 54: Compare extended Random Forest with station baseline

print("\nTEST 54: Random Forest vs station baseline")

# Historical good-week probability for each station
station_probabilities = extended_train.groupby(
    "station_code"
)["good_week"].mean()

# Assign historical probabilities to validation examples
baseline_prob = extended_val["station_code"].map(
    station_probabilities
)

# Turn probabilities into good/bad predictions
baseline_pred = (baseline_prob >= 0.5).astype(int)

# Compare the two methods
print("Baseline Accuracy:",
      round(accuracy_score(y_all_val, baseline_pred) * 100, 2), "%")

print("Random Forest Accuracy:",
      round(accuracy_score(y_all_val, all_pred) * 100, 2), "%")

print("Baseline Brier Score:",
      round(brier_score_loss(y_all_val, baseline_prob), 4))

print("Random Forest Brier Score:",
      round(brier_score_loss(y_all_val, all_prob), 4))


# Test 55: Compare models on additional stations only

print("\nTEST 55: New stations - baseline vs Random Forest")

# Select validation examples from the additional stations
new_station_mask = ~extended_val["station_code"].isin(original_codes)

# Actual results for these examples
y_new = y_all_val[new_station_mask]

# Baseline predictions
baseline_new_prob = baseline_prob[new_station_mask]
baseline_new_pred = baseline_pred[new_station_mask]

# Random Forest predictions
rf_new_prob = all_prob[new_station_mask.to_numpy()]
rf_new_pred = all_pred[new_station_mask.to_numpy()]

print("Examples:", len(y_new))
print("Stations:", extended_val.loc[new_station_mask, "station_code"].nunique())

print("\nBaseline")
print("Accuracy:", round(accuracy_score(y_new, baseline_new_pred) * 100, 2), "%")
print("Brier:", round(brier_score_loss(y_new, baseline_new_prob), 4))

print("\nRandom Forest")
print("Accuracy:", round(accuracy_score(y_new, rf_new_pred) * 100, 2), "%")
print("Brier:", round(brier_score_loss(y_new, rf_new_prob), 4))


# Test 56: Random Forest with station identity

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

print("\nTEST 56: Random Forest with station identity")

# Our new input features
new_features = [
    "altitude_m",
    "month",
    "day",
    "station_code"
]

# Prepare training and validation inputs
X_train_56 = extended_train[new_features]
y_train_56 = extended_train["good_week"].astype(int)

X_val_56 = extended_val[new_features]
y_val_56 = extended_val["good_week"].astype(int)

# Convert station codes into separate 0/1 columns
preprocessor_56 = ColumnTransformer(
    transformers=[
        (
            "station",
            OneHotEncoder(handle_unknown="ignore"),
            ["station_code"]
        )
    ],
    remainder="passthrough"
)

# Create our Random Forest with the same settings
rf_station = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=20,
    random_state=42
)

# Combine preprocessing and Random Forest
station_pipeline = Pipeline([
    ("preprocessing", preprocessor_56),
    ("model", rf_station)
])

# Train the new model
station_pipeline.fit(X_train_56, y_train_56)

# Make predictions
station_prob = station_pipeline.predict_proba(X_val_56)[:, 1]
station_pred = station_pipeline.predict(X_val_56)

# Evaluate the model
print("Accuracy:",
      round(accuracy_score(y_val_56, station_pred) * 100, 2), "%")

print("Brier Score:",
      round(brier_score_loss(y_val_56, station_prob), 4))

print("Training stations:",
      extended_train["station_code"].nunique())


# Test 57: Seasonal historical baseline

print("\nTEST 57: Seasonal historical baseline")

# Calculate historical probability per station and month
monthly_probabilities = extended_train.groupby(
    ["station_code", "month"]
)["good_week"].mean()

# Calculate historical probability per station
station_probabilities_57 = extended_train.groupby(
    "station_code"
)["good_week"].mean()

# Prepare validation data
seasonal_val = extended_val.copy()

# Assign historical probabilities for matching station and month
seasonal_val["seasonal_prob"] = [
    monthly_probabilities.get(
        (row.station_code, row.month),
        station_probabilities_57[row.station_code]
    )
    for row in seasonal_val.itertuples()
]

# Convert probabilities to classifications
seasonal_pred = (
    seasonal_val["seasonal_prob"] >= 0.5
).astype(int)

# Compare predictions with actual results
print("Validation examples:", len(seasonal_val))

print("Seasonal Baseline Accuracy:",
      round(accuracy_score(
          seasonal_val["good_week"].astype(int),
          seasonal_pred
      ) * 100, 2), "%")

print("Seasonal Baseline Brier Score:",
      round(brier_score_loss(
          seasonal_val["good_week"].astype(int),
          seasonal_val["seasonal_prob"]
      ), 4))

print("\nRandom Forest with station identity")
print("Accuracy:",
      round(accuracy_score(y_val_56, station_pred) * 100, 2), "%")

print("Brier Score:",
      round(brier_score_loss(y_val_56, station_prob), 4))


# Test 58: Compare Random Forest models per station

print("\nTEST 58: Improvement from station identity")

comparison_58 = extended_val.copy()

# Predictions from old and new Random Forest
comparison_58["old_prediction"] = all_pred
comparison_58["new_prediction"] = station_pred

for code, data in comparison_58.groupby("station_code"):

    actual = data["good_week"].astype(int)

    old_accuracy = accuracy_score(
        actual, data["old_prediction"]
    )

    new_accuracy = accuracy_score(
        actual, data["new_prediction"]
    )

    improvement = (new_accuracy - old_accuracy) * 100

    print(
        code,
        "| Old:", round(old_accuracy * 100, 1), "%",
        "| New:", round(new_accuracy * 100, 1), "%",
        "| Change:", round(improvement, 1),
        "percentage points"
    )


# Test 59: Understand predictions at selected stations

print("\nTEST 59: Station predictions and actual snow conditions")

# Add predictions to validation data
analysis_59 = extended_val.copy()
analysis_59["rf_prediction"] = station_pred
analysis_59["rf_probability"] = station_prob

# Investigate stations with interesting results
selected_stations = ["ZER", "PON", "DAV", "MVE"]

for code in selected_stations:

    train_station = extended_train[
        extended_train["station_code"] == code
    ]

    val_station = analysis_59[
        analysis_59["station_code"] == code
    ]

    print("\nStation:", code)

    print(
        "Training: Actual good weeks:",
        round(train_station["good_week"].mean() * 100, 1), "%"
    )

    print(
        "Validation: Actual good weeks:",
        round(val_station["good_week"].mean() * 100, 1), "%"
    )

    print(
        "Validation: Predicted good weeks:",
        round(val_station["rf_prediction"].mean() * 100, 1), "%"
    )

    print(
        "Validation: Average predicted probability:",
        round(val_station["rf_probability"].mean() * 100, 1), "%"
    )

    print(
        "Validation Brier Score:",
        round(
            brier_score_loss(
                val_station["good_week"].astype(int),
                val_station["rf_probability"]
            ),
            4
        )
    )


# Test 60: Random Forest vs baseline at selected stations

print("\nTEST 60: Station baseline vs Random Forest")

# Use the same validation data and predictions
comparison_60 = extended_val.copy()

comparison_60["rf_prob"] = station_prob
comparison_60["baseline_prob"] = baseline_prob

selected_stations = ["ZER", "PON", "DAV", "MVE"]

for code in selected_stations:

    data = comparison_60[
        comparison_60["station_code"] == code
    ]

    actual = data["good_week"].astype(int)

    baseline_brier = brier_score_loss(
        actual,
        data["baseline_prob"]
    )

    rf_brier = brier_score_loss(
        actual,
        data["rf_prob"]
    )

    print("\nStation:", code)
    print("Baseline Brier:", round(baseline_brier, 4))
    print("Random Forest Brier:", round(rf_brier, 4))

    if rf_brier < baseline_brier:
        print("Better model: Random Forest")
    elif baseline_brier < rf_brier:
        print("Better model: Baseline")
    else:
        print("Both models are equal")


# Test 61: Compare Brier Scores across all stations

print("\nTEST 61: Brier Score comparison for all stations")

comparison_61 = extended_val.copy()

comparison_61["rf_prob"] = station_prob
comparison_61["baseline_prob"] = baseline_prob

rf_wins = 0
baseline_wins = 0

for code, data in comparison_61.groupby("station_code"):

    actual = data["good_week"].astype(int)

    rf_brier = brier_score_loss(
        actual,
        data["rf_prob"]
    )

    baseline_brier = brier_score_loss(
        actual,
        data["baseline_prob"]
    )

    difference = baseline_brier - rf_brier

    if difference > 0:
        winner = "Random Forest"
        rf_wins += 1
    elif difference < 0:
        winner = "Baseline"
        baseline_wins += 1
    else:
        winner = "Tie"

    print(
        code,
        "| Baseline:", round(baseline_brier, 4),
        "| RF:", round(rf_brier, 4),
        "| Difference:", round(difference, 4),
        "| Winner:", winner
    )

print("\nRandom Forest wins:", rf_wins)
print("Baseline wins:", baseline_wins)


# Test 62: Detailed data quality check for FIO

print("\nTEST 62: Detailed snow data quality for FIO")

fio_snow = load_snow_depth(
    "FIO", "2015-01-01", "2023-12-31"
)

fio_snow = fio_snow.sort_values("date").copy()

print("Total rows:", len(fio_snow))

print(
    "Valid snow measurements:",
    fio_snow["snow_depth_cm"].notna().sum()
)

print(
    "Missing snow measurements:",
    fio_snow["snow_depth_cm"].isna().sum()
)

print(
    "Duplicate dates:",
    fio_snow["date"].duplicated().sum()
)

# Detect calendar gaps
date_gaps = fio_snow["date"].diff().dropna()

print(
    "Gaps longer than one day:",
    (date_gaps > pd.Timedelta(days=1)).sum()
)

# Check valid weeks by year
fio_weeks = label_good_weeks(fio_snow)

valid_fio_weeks = fio_weeks.dropna(
    subset=["good_week"]
)

print("\nValid weeks per year:")
print(valid_fio_weeks.groupby("year").size())

# Show daily coverage and missing values by year
fio_snow["year"] = fio_snow["date"].dt.year

print("\nMeasurements per year:")
print(
    fio_snow.groupby("year")["snow_depth_cm"]
    .agg(
        total_days="size",
        valid_days="count"
    )
)


# Test 63: Check missing measurements by month

print("\nTEST 63: Missing snow measurements by month")

# Use the FIO data already loaded in Test 62
monthly_quality = fio_snow.copy()

# Extract month from each date
monthly_quality["month"] = monthly_quality["date"].dt.month

# Count valid and missing measurements per month
monthly_summary = monthly_quality.groupby("month")[
    "snow_depth_cm"
].agg(
    total_days="size",
    valid_days="count"
)

monthly_summary["missing_days"] = (
    monthly_summary["total_days"] -
    monthly_summary["valid_days"]
)

monthly_summary["coverage_percent"] = (
    monthly_summary["valid_days"] /
    monthly_summary["total_days"] * 100
).round(1)

print(monthly_summary)


# Test 64: Preserve average snow depth for each week

print("\nTEST 64: Average snow depth per week")

def label_weeks_with_snow_depth(snow_data):

    snow_data = snow_data.sort_values(
        ["station_code", "date"]
    )

    results = []

    for station, station_data in snow_data.groupby("station_code"):

        station_data = station_data.reset_index(drop=True)

        for i in range(len(station_data) - 6):

            week_data = station_data.iloc[i:i+7]

            # Check for seven consecutive calendar days
            date_differences = (
                week_data["date"].diff().dropna()
            )

            if not (
                date_differences == pd.Timedelta(days=1)
            ).all():
                continue

            # Only use weeks fully within December-April
            if not week_data["date"].dt.month.isin(
                [12, 1, 2, 3, 4]
            ).all():
                continue

            snow_values = week_data["snow_depth_cm"]

            # Do not calculate averages with missing values
            if snow_values.isna().any():
                average_snow = np.nan
                good_week = np.nan
            else:
                average_snow = snow_values.mean()
                good_week = int(average_snow >= 50)

            results.append({
                "station_code": station,
                "week_start": week_data["date"].iloc[0],
                "average_snow_depth_cm": average_snow,
                "good_week": good_week
            })

    return pd.DataFrame(results)


# Check the function using real Davos measurements
snow_64 = load_snow_depth(
    "DAV", "2020-02-01", "2020-02-29"
)

labels_64 = label_weeks_with_snow_depth(snow_64)

print(labels_64.head(10).to_string(index=False))

print("\nValid weeks:", labels_64["good_week"].notna().sum())

print(
    "Missing snow averages:",
    labels_64["average_snow_depth_cm"].isna().sum()
)

print(
    "Average snow depth range:",
    round(labels_64["average_snow_depth_cm"].min(), 1),
    "to",
    round(labels_64["average_snow_depth_cm"].max(), 1),
    "cm"
)

# Compare with our original label function
old_labels_64 = label_good_weeks(snow_64)

same_labels = (
    labels_64["good_week"].fillna(-1).reset_index(drop=True)
    ==
    old_labels_64["good_week"].fillna(-1).reset_index(drop=True)
).all()

print("Labels match original function:", same_labels)


# Test 65: Historical monthly snow depth without future data

print("\nTEST 65: Historical monthly snow depth")

# Load historical daily snow measurements for Davos
snow_65 = load_snow_depth(
    "DAV", "2015-01-01", "2023-12-31"
)

# Keep February measurements only
february_65 = snow_65[
    snow_65["date"].dt.month == 2
].copy()

# Extract year
february_65["year"] = february_65["date"].dt.year

# Calculate average February snow depth for each year
yearly_february = february_65.groupby(
    "year"
)["snow_depth_cm"].mean()

print("\nAverage February snow depth per year:")
print(yearly_february.round(1))

# For each year, calculate the average using
# only the previous years
for prediction_year in range(2016, 2024):

    previous_years = yearly_february[
        yearly_february.index < prediction_year
    ]

    if len(previous_years) == 0:
        continue

    historical_average_cm = previous_years.mean()

    print(
        "Prediction year:", prediction_year,
        "| Historical February average:",
        round(historical_average_cm, 1), "cm",
        "| Years used:",
        list(previous_years.index)
    )


# Test 66: Add historical snow depth to Random Forest

print("\nTEST 66: Random Forest with historical snow depth")

# STEP 1: Load historical snow measurements (2015-2021)
monthly_data_66 = []

for code in sorted(extended_train["station_code"].unique()):

    snow = load_snow_depth(
        code, "2015-01-01", "2021-12-31"
    )

    snow = snow.copy()
    snow["year"] = snow["date"].dt.year
    snow["month"] = snow["date"].dt.month

    # One average per station, year and month
    monthly = (
        snow.groupby(["year", "month"])["snow_depth_cm"]
        .mean()
        .dropna()
        .reset_index()
    )

    monthly["station_code"] = code
    monthly_data_66.append(monthly)

monthly_snow_66 = pd.concat(
    monthly_data_66, ignore_index=True
)

print("Historical monthly records:", len(monthly_snow_66))


# STEP 2: Calculate historical features without leakage

history_lookup_66 = {}

for code in monthly_snow_66["station_code"].unique():

    for month in [12, 1, 2, 3, 4]:

        station_month = monthly_snow_66[
            (monthly_snow_66["station_code"] == code) &
            (monthly_snow_66["month"] == month)
        ]

        for prediction_year in range(2015, 2024):

            # Validation uses only training years
            cutoff = min(prediction_year, 2022)

            past = station_month[
                station_month["year"] < cutoff
            ]

            if len(past) > 0:
                historical_cm = past["snow_depth_cm"].mean()
            else:
                historical_cm = np.nan

            history_lookup_66[
                (code, month, prediction_year)
            ] = historical_cm


# STEP 3: Add feature to training and validation data

train_66 = extended_train.copy()
val_66 = extended_val.copy()

for dataset in [train_66, val_66]:

    dataset["historical_snow_depth_cm"] = [
        history_lookup_66.get(
            (row.station_code, row.month, row.year),
            np.nan
        )
        for row in dataset.itertuples()
    ]

# Remove examples without historical information
train_66 = train_66.dropna(
    subset=["historical_snow_depth_cm"]
)

val_66 = val_66.dropna(
    subset=["historical_snow_depth_cm"]
)

print("Training examples:", len(train_66))
print("Validation examples:", len(val_66))

print("\nExample historical features:")
print(
    val_66[
        (val_66["station_code"] == "DAV") &
        (val_66["month"] == 2)
    ][
        ["station_code", "week_start",
         "historical_snow_depth_cm"]
    ].head(3).to_string(index=False)
)


# STEP 4: Compare both models on the same data

old_features_66 = [
    "altitude_m", "month", "day", "station_code"
]

new_features_66 = old_features_66 + [
    "historical_snow_depth_cm"
]

y_train_66 = train_66["good_week"].astype(int)
y_val_66 = val_66["good_week"].astype(int)

def create_model_66():

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

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", forest)
    ])


# Old model: without historical snow depth
old_model_66 = create_model_66()

old_model_66.fit(
    train_66[old_features_66],
    y_train_66
)

old_prob_66 = old_model_66.predict_proba(
    val_66[old_features_66]
)[:, 1]


# New model: with historical snow depth
new_model_66 = create_model_66()

new_model_66.fit(
    train_66[new_features_66],
    y_train_66
)

new_prob_66 = new_model_66.predict_proba(
    val_66[new_features_66]
)[:, 1]


# STEP 5: Evaluate both models

for name, probabilities in [
    ("Old RF", old_prob_66),
    ("New RF", new_prob_66)
]:

    predictions = (probabilities >= 0.5).astype(int)

    print("\n", name)
    print(
        "Accuracy:",
        round(accuracy_score(y_val_66, predictions) * 100, 2),
        "%"
    )
    print(
        "Brier Score:",
        round(brier_score_loss(y_val_66, probabilities), 4)
    )

print(
    "\nBrier improvement:",
    round(
        brier_score_loss(y_val_66, old_prob_66) -
        brier_score_loss(y_val_66, new_prob_66),
        4
    )
)


# Test 67: Compare model probabilities per station

print("\nTEST 67: Model comparison per station")

comparison_67 = val_66.copy()

comparison_67["old_rf_prob"] = old_prob_66
comparison_67["new_rf_prob"] = new_prob_66

# Baseline trained on the same training examples
station_baseline_67 = train_66.groupby(
    "station_code"
)["good_week"].mean()

comparison_67["baseline_prob"] = (
    comparison_67["station_code"]
    .map(station_baseline_67)
)

print("\nBrier Scores by station:")

new_rf_wins = 0
old_rf_wins = 0

for code, data in comparison_67.groupby("station_code"):

    actual = data["good_week"].astype(int)

    old_brier = brier_score_loss(
        actual, data["old_rf_prob"]
    )

    new_brier = brier_score_loss(
        actual, data["new_rf_prob"]
    )

    baseline_brier = brier_score_loss(
        actual, data["baseline_prob"]
    )

    if new_brier < old_brier:
        winner = "New RF"
        new_rf_wins += 1
    elif old_brier < new_brier:
        winner = "Old RF"
        old_rf_wins += 1
    else:
        winner = "Tie"

    print(
        code,
        "| Old RF:", round(old_brier, 4),
        "| New RF:", round(new_brier, 4),
        "| Baseline:", round(baseline_brier, 4),
        "| Winner:", winner
    )

print("\nNew RF wins:", new_rf_wins)
print("Old RF wins:", old_rf_wins)

print("\nOverall baseline Brier:",
      round(brier_score_loss(
          comparison_67["good_week"].astype(int),
          comparison_67["baseline_prob"]
      ), 4))


# Test 68: Model performance by validation year

print("\nTEST 68: Random Forest performance by year")

results_68 = extended_val.copy()

# Predictions from Test 56
results_68["rf_probability"] = station_prob
results_68["rf_prediction"] = station_pred

# Historical baseline from Test 54
results_68["baseline_probability"] = baseline_prob

for year, data in results_68.groupby("year"):

    actual = data["good_week"].astype(int)

    rf_accuracy = accuracy_score(
        actual, data["rf_prediction"]
    )

    rf_brier = brier_score_loss(
        actual, data["rf_probability"]
    )

    baseline_brier = brier_score_loss(
        actual, data["baseline_probability"]
    )

    print("\nYear:", year)
    print("Examples:", len(data))
    print("Actual good weeks:", round(actual.mean() * 100, 1), "%")
    print("RF Accuracy:", round(rf_accuracy * 100, 2), "%")
    print("RF Brier Score:", round(rf_brier, 4))
    print("Baseline Brier Score:", round(baseline_brier, 4))


# Test 69: Check whether our model overestimates snow reliability

print("\nTEST 69: Predicted vs actual good weeks by year")

results_69 = extended_val.copy()

# Probabilities from our Random Forest in Test 56
results_69["rf_probability"] = station_prob

# Historical probabilities from the station baseline
results_69["baseline_probability"] = baseline_prob

for year, data in results_69.groupby("year"):

    actual_good = data["good_week"].mean() * 100

    rf_expected = data["rf_probability"].mean() * 100

    baseline_expected = (
        data["baseline_probability"].mean() * 100
    )

    print("\nYear:", year)

    print(
        "Actually good weeks:",
        round(actual_good, 2), "%"
    )

    print(
        "RF average predicted probability:",
        round(rf_expected, 2), "%"
    )

    print(
        "Baseline average predicted probability:",
        round(baseline_expected, 2), "%"
    )

    print(
        "RF overestimation:",
        round(rf_expected - actual_good, 2),
        "percentage points"
    )


# Test 70: Walk-forward validation across four years

print("\nTEST 70: Walk-forward validation")

features_70 = [
    "altitude_m", "month", "day", "station_code"
]

for test_year in [2020, 2021, 2022, 2023]:

    year_start = pd.Timestamp(f"{test_year}-01-01")
    next_year = pd.Timestamp(f"{test_year + 1}-01-01")

    # Training weeks must end before the test year
    train_70 = extended_ml_data[
        extended_ml_data["week_start"] +
        pd.Timedelta(days=6) < year_start
    ].copy()

    # Test weeks must be fully within the test year
    test_70 = extended_ml_data[
        (extended_ml_data["week_start"] >= year_start) &
        (
            extended_ml_data["week_start"] +
            pd.Timedelta(days=6) < next_year
        )
    ].copy()

    # Only evaluate stations present in training
    test_70 = test_70[
        test_70["station_code"].isin(
            train_70["station_code"].unique()
        )
    ].copy()

    X_train = train_70[features_70]
    y_train = train_70["good_week"].astype(int)

    X_test = test_70[features_70]
    y_test = test_70["good_week"].astype(int)

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

    model_70 = Pipeline([
        ("preprocessing", preprocessing),
        ("model", forest)
    ])

    model_70.fit(X_train, y_train)

    rf_prob = model_70.predict_proba(X_test)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)

    # Historical baseline from training years only
    station_history = train_70.groupby(
        "station_code"
    )["good_week"].mean()

    baseline_prob = test_70["station_code"].map(
        station_history
    )

    print("\nYear:", test_year)
    print("Training examples:", len(train_70))
    print("Test examples:", len(test_70))

    print(
        "RF Accuracy:",
        round(accuracy_score(y_test, rf_pred) * 100, 2),
        "%"
    )

    print(
        "RF Brier:",
        round(brier_score_loss(y_test, rf_prob), 4)
    )

    print(
        "Baseline Brier:",
        round(brier_score_loss(y_test, baseline_prob), 4)
    )

    print(
        "Actual good weeks:",
        round(y_test.mean() * 100, 2),
        "%"
    )


# Test 71: Walk-forward validation with monthly baseline

print("\nTEST 71: Monthly baseline vs Random Forest")

for test_year in [2020, 2021, 2022, 2023]:

    year_start = pd.Timestamp(f"{test_year}-01-01")
    next_year = pd.Timestamp(f"{test_year + 1}-01-01")

    # Same training and test periods as Test 70
    train_71 = extended_ml_data[
        extended_ml_data["week_start"] +
        pd.Timedelta(days=6) < year_start
    ].copy()

    test_71 = extended_ml_data[
        (extended_ml_data["week_start"] >= year_start) &
        (
            extended_ml_data["week_start"] +
            pd.Timedelta(days=6) < next_year
        )
    ].copy()

    test_71 = test_71[
        test_71["station_code"].isin(
            train_71["station_code"].unique()
        )
    ].copy()

    # Historical probability for each station and month
    monthly_history = train_71.groupby(
        ["station_code", "month"]
    )["good_week"].mean()

    # Fallback: historical probability for the station
    station_history = train_71.groupby(
        "station_code"
    )["good_week"].mean()

    monthly_probabilities = [
        monthly_history.get(
            (row.station_code, row.month),
            station_history[row.station_code]
        )
        for row in test_71.itertuples()
    ]

    y_test_71 = test_71["good_week"].astype(int)

    monthly_brier = brier_score_loss(
        y_test_71, monthly_probabilities
    )

    print("\nYear:", test_year)
    print("Test examples:", len(test_71))
    print("Monthly Baseline Brier:", round(monthly_brier, 4))


# Test 72: Historical baseline with +/- 14-day window

print("\nTEST 72: Rolling 14-day baseline")

for test_year in [2020, 2021, 2022, 2023]:

    year_start = pd.Timestamp(f"{test_year}-01-01")
    next_year = pd.Timestamp(f"{test_year + 1}-01-01")

    # Use the same periods as Test 70 and 71
    train_72 = extended_ml_data[
        extended_ml_data["week_start"] +
        pd.Timedelta(days=6) < year_start
    ].copy()

    test_72 = extended_ml_data[
        (extended_ml_data["week_start"] >= year_start) &
        (extended_ml_data["week_start"] +
         pd.Timedelta(days=6) < next_year)
    ].copy()

    test_72 = test_72[
        test_72["station_code"].isin(
            train_72["station_code"].unique()
        )
    ].copy()

    # Put dates onto the same reference calendar.
    # Year 2000 is used to handle leap days consistently.
    for dataset in [train_72, test_72]:
        dataset["calendar_day"] = pd.to_datetime(
            "2000-" +
            dataset["month"].astype(str).str.zfill(2) +
            "-" +
            dataset["day"].astype(str).str.zfill(2)
        ).dt.dayofyear

    # Fallback probability for each station
    station_history = train_72.groupby(
        "station_code"
    )["good_week"].mean()

    rolling_prob = pd.Series(
        index=test_72.index, dtype=float
    )

    # Calculate probabilities from nearby calendar dates
    for code, station_test in test_72.groupby("station_code"):

        station_train = train_72[
            train_72["station_code"] == code
        ]

        for day, rows in station_test.groupby("calendar_day"):

            distance = abs(
                station_train["calendar_day"] - day
            )

            # Also handle dates around New Year
            distance = np.minimum(distance, 366 - distance)

            nearby = station_train[
                distance <= 14
            ]

            if len(nearby) > 0:
                probability = nearby["good_week"].mean()
            else:
                probability = station_history[code]

            rolling_prob.loc[rows.index] = probability

    y_test = test_72["good_week"].astype(int)

    rolling_brier = brier_score_loss(
        y_test, rolling_prob
    )

    print("\nYear:", test_year)
    print("Test examples:", len(test_72))
    print("Rolling Baseline Brier:", round(rolling_brier, 4))


# Test 73: Save our existing ML dataset

print("\nTEST 73: Save ML dataset")

# Save the exact dataset prepared in Test 47
extended_ml_data.to_csv(
    "ml_dataset.csv",
    index=False
)

print("ML dataset saved successfully!")
print("Rows:", len(extended_ml_data))
print("Stations:", extended_ml_data["station_code"].nunique())
print("Columns:", extended_ml_data.columns.tolist())

