"""Snow-Proof: schlanke ML-Pipeline (separat vom Experimentarchiv).

Benötigt die bestehende data.py im selben Projektordner.
Train: Wochen mit Ende bis 31.12.2021; Validierung: 2022–2023.
2024 wird absichtlich nicht verwendet (späterer unabhängiger Test).
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ["altitude_m", "month", "day", "station_code"]
MONTHS = (12, 1, 2, 3, 4)
TRAIN_CUTOFF = pd.Timestamp("2022-01-01")
VAL_CUTOFF = pd.Timestamp("2024-01-01")


def label_good_weeks(snow_data):
    """Label = 1, wenn Ø Schneehöhe über 7 *aufeinanderfolgenden* Tagen >=50 cm.

    Bei fehlenden Messungen ist das Label unbekannt (NaN).
    Die Ausgabe enthält auch die Wochenmittelwerte, aber diese sind NICHT
    als Input für die Vorhersage derselben Woche zulässig (Data Leakage).
    """
    required = {"station_code", "date", "snow_depth_cm"}
    missing = required - set(snow_data.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten in snow_data: {sorted(missing)}")

    data = snow_data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["station_code", "date"])
    records = []
    for station, group in data.groupby("station_code"):
        group = group.reset_index(drop=True)
        for i in range(len(group) - 6):
            week = group.iloc[i:i + 7]
            dates = week["date"]
            if not dates.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
                continue
            if not dates.dt.month.isin(MONTHS).all():
                continue
            measurements = week["snow_depth_cm"]
            mean_cm = float(measurements.mean()) if measurements.notna().all() else np.nan
            records.append({
                "station_code": station,
                "week_start": dates.iloc[0],
                "week_end": dates.iloc[-1],
                "good_week": int(mean_cm >= 50) if pd.notna(mean_cm) else np.nan,
                "average_snow_depth_cm": mean_cm,
            })
    return pd.DataFrame(records, columns=[
        "station_code", "week_start", "week_end",
        "good_week", "average_snow_depth_cm"
    ])


def prepare_dataset(start="2015-01-01", end="2023-12-31"):
    """Lädt die stationsbezogenen Daten über die vorhandene data.py.

    Nicht unterstützte Resorts und Stationen ohne gültige Wochen werden
    ausgelassen; die Stationsauswahl bleibt die bestehende Zuordnung aus data.py.
    """
    from data import load_resorts, load_snow_depth, load_stations, link_stations_to_resorts

    stations = load_stations()
    resorts = load_resorts()
    linked = link_stations_to_resorts(resorts, stations)
    codes = sorted(linked["station_code"].dropna().unique())

    frames = []
    ignored = []
    for code in codes:
        snow = load_snow_depth(code, start, end)
        if snow.empty:
            ignored.append(code)
            continue
        labels = label_good_weeks(snow).dropna(subset=["good_week"])
        if labels.empty:
            ignored.append(code)
            continue
        frames.append(labels)

    if not frames:
        raise ValueError("Keine gültigen Wochen gefunden. Bitte Datenbank/API prüfen.")

    ml_data = pd.concat(frames, ignore_index=True)
    ml_data = ml_data.merge(
        stations[["station_code", "altitude_m"]],
        on="station_code", how="inner", validate="many_to_one"
    )
    ml_data["month"] = ml_data["week_start"].dt.month
    ml_data["day"] = ml_data["week_start"].dt.day
    ml_data["year"] = ml_data["week_start"].dt.year
    ml_data = ml_data.dropna(subset=["altitude_m"])

    # Letzte 6 Tage vor dem 1. Januar 2024 bilden keine volle Woche im
    # verfügbaren Datensatz und dürfen nicht als gültige Woche gelten.
    ml_data = ml_data[ml_data["week_end"] < VAL_CUTOFF].copy()

    print(f"Stationen: {ml_data['station_code'].nunique()}; "
          f"gültige Wochen: {len(ml_data)}")
    print(f"Stationen ohne gültige Daten: {ignored}")
    return ml_data


def split_dataset(data):
    train = data[data["week_end"] < TRAIN_CUTOFF].copy()
    val = data[(data["week_start"] >= TRAIN_CUTOFF) &
               (data["week_end"] < VAL_CUTOFF)].copy()
    assert train["week_end"].max() < val["week_start"].min(), "Zeitliche Überlappung!"
    return train, val


def create_random_forest(features=FEATURES):
    preprocessor = ColumnTransformer(
        transformers=[("station", OneHotEncoder(handle_unknown="ignore"), ["station_code"])],
        remainder="passthrough"
    )
    classifier = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=20, random_state=42
    )
    return Pipeline([("preprocessing", preprocessor), ("model", classifier)])


def station_baseline(train, val):
    probabilities = train.groupby("station_code")["good_week"].mean()
    # Falls eine Validierungsstation zuvor nicht trainiert wurde:
    fallback = float(train["good_week"].mean())
    return val["station_code"].map(probabilities).fillna(fallback).to_numpy()


def monthly_baseline(train, val):
    monthly = train.groupby(["station_code", "month"])["good_week"].mean()
    station = train.groupby("station_code")["good_week"].mean()
    fallback = float(train["good_week"].mean())
    return np.array([
        monthly.get((row.station_code, row.month),
                    station.get(row.station_code, fallback))
        for row in val.itertuples()
    ])
