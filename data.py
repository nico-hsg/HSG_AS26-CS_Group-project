"""
Snow-Proof: getting the data (Section 1, Nico).

Loads MeteoSwiss stations and snow depth, and stores them in a SQLite database
(snowproof.db), so data is only downloaded when needed.
streamlit_app.py imports the functions from here.
"""

import streamlit as st
import pandas as pd
import sqlite3 as sql
import requests

# =============================================================================
# SECTION 1: GETTING THE DATA (Nico, Task 3 and 5)
# Website part: nothing visible directly. Everything else is built on this data.
# =============================================================================

DB_PATH = "snowproof.db"  # Database file, created in the project folder.
RESORTS_PATH = "resorts.csv"  # Hand-made resort list, in the project folder.
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"  # Open-Meteo historical weather (ERA5), from 1940.

def init_database():
    """
    Goal:       Create the SQL database, covering data needs of subsequent functions.
    Website:    Not visible. Stores data.
    Returns:    Nothing. Creates the tables in snowproof.db if they don't exist yet.
    """
    conn = sql.connect(DB_PATH)

    # Create snow_depth_table, unless already existing.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snow_depth_table (
            station_code TEXT,
            date TEXT,
            snow_depth_cm REAL,
            PRIMARY KEY (station_code, date)
        )
    """)

    # Create update table, unless already existing.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS update_table (
            dataset TEXT,
            location TEXT,
            download_date TEXT,
            PRIMARY KEY (dataset, location)
        )
    """)

    conn.commit()
    conn.close()


# =============================================================================
# Helper functions
# =============================================================================

def save_snow_depth(snow):
    """
    Goal:       Save a snow depth table into snow_depth_table in SQLite.
    Input:      Snow (a table with station_code, date, snow_depth_cm (as returned by load_snow_depth)).
    Website:    Not visible. Stores data for later use.
    Returns:    Nothing. Writes the rows into snowproof.db (replaces days that already exist).
    """
    snow = snow.copy()  # Copy to not change caller's table
    snow["date"] = snow["date"].dt.strftime("%Y-%m-%d")  # Turn real dates into text dates
    rows = snow.itertuples(index=False, name=None)  # Turn table into list of rows
    conn = sql.connect(DB_PATH)
    conn.executemany("INSERT OR REPLACE INTO snow_depth_table VALUES (?, ?, ?)", rows)  # Insert rows into snow_depth_table
    conn.commit()
    conn.close()


def read_snow_depth(station_code, start_date, end_date):
    """
    Goal:       Read a snow depth table from SQLite.
    Input:      Station_code, start_date and end_date
    Website:    Not visible.
    Returns:    A table with station_code, date, snow_depth_cm.
    """
    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")  # Turn real dates into text dates
    end = pd.to_datetime(end_date).strftime("%Y-%m-%d")  # Turn real dates into text dates
    conn = sql.connect(DB_PATH)

    # Filter and sort text dates table
    result = pd.read_sql_query(
        "SELECT station_code, date, snow_depth_cm FROM snow_depth_table WHERE station_code = ? AND date BETWEEN ? AND ? ORDER BY date",
        conn,
        params=(station_code, start, end),
    )
    conn.close()

    result["date"] = pd.to_datetime(result["date"], format="%Y-%m-%d")  # Turn text dates into real dates
    return result


def mark_downloaded(dataset, location):
    """
    Goal:       Logs the last download date of a dataset and location combination.
    Input:      Dataset and location.
    Website:    Not visible. Remembers when we last downloaded a file.
    Returns:    Nothing. Writes one row into update_table.
    """
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")  # The current date and time as text

    # Write it into the logbook (replaces the old time for the same file)
    conn = sql.connect(DB_PATH)
    conn.execute("INSERT OR REPLACE INTO update_table VALUES (?, ?, ?)", (dataset, location, now))
    conn.commit()
    conn.close()


def needs_download(dataset, location):
    """
    Goal:       Determine whether data needs a fresh download or not.
    Input:      Dataset and location.
    Website:    Not visible. Decides if we must download a file again.
    Returns:    True (download it) or False (use the saved data).
    """
    # Look up in the logbook when this file was last downloaded
    conn = sql.connect(DB_PATH)
    row = conn.execute(
        "SELECT download_date FROM update_table WHERE dataset = ? AND location = ?",
        (dataset, location),
    ).fetchone()
    conn.close()

    # Never downloaded before = download
    if row is None:
        return True

    last_download = row[0]
    today = pd.Timestamp.now().strftime("%Y-%m-%d")

    # Recent file: new data every day = download if the last download was before today
    if dataset == "snow_recent":
        return last_download[:10] < today

    # Historical file: changes once a year = download if the last download was in an earlier year
    return last_download[:4] < today[:4]


def download_snow_file(url):
    """
    Goal:       Download one MeteoSwiss snow file and prepare it
    Input:      URL of the CSV file (historical or recent)
    Website:    Not visible.
    Returns:    A table with station_code, date, snow_depth_cm (all days in the file).
    """
    # Reading one MeteoSwiss file (historical or recent)
    snow = pd.read_csv(url, sep=";", encoding="cp1252")

    # Converting text dates into real dates
    snow["date"] = pd.to_datetime(snow["reference_timestamp"], format="%d.%m.%Y %H:%M")

    # Renaming and keeping only the target columns
    snow = snow.rename(columns={"station_abbr": "station_code",
                                          "hto000d0": "snow_depth_cm"
                                          })
    snow = snow[["station_code", "date", "snow_depth_cm"]].reset_index(drop=True)

    return snow


# Rule for linking a snow station to a resort:
# a station counts if it is at most MAX_STATION_DISTANCE_KM away and at most
# MAX_STATION_BELOW_BASE_M below the resort's lowest point.
#  If that applies to multiple stations, the nearest one (horizontal distance) wins.
MAX_STATION_DISTANCE_KM = 15
MAX_STATION_BELOW_BASE_M = 200

KM_PER_LAT = 111  # km per degree of latitude (north-south)
KM_PER_LON = 76  # km per degree of longitude at Swiss latitudes (east-west)


def measure_distance_km(lat1, lon1, lat2, lon2):
    """
    Goal:       Measure distance between two geographical points.
    Input:      2 set of latitude and longitude (lat1, lon1, lat2, lon2)
    Website:    Not visible. Background calculation.
    Returns:    A distance value in km.
    """
    north_south_km = abs(lat2 - lat1) * KM_PER_LAT
    east_west_km = abs(lon2 - lon1) * KM_PER_LON

    return ((north_south_km ** 2) + (east_west_km ** 2)) ** (1/2)  # Using pythagoras


# =============================================================================
# Load functions
# =============================================================================

@st.cache_data
def load_stations():
    """
    Goal:       A list of the MeteoSwiss stations that measure snow depth, with their position and altitude.
    Website:    Not visible. Decides which resorts can get a snow forecast.
    Source:     MeteoSwiss Open Data (ch.meteoschweiz.ogd-nime).
    Returns:    A table with station_code, station_name, lat, lon, and altitude_m.
    """

    # Loading weather station data from Meteoschweiz.
    stations = pd.read_csv(
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nime/ogd-nime_meta_stations.csv",
        sep=";",
        encoding="cp1252"
    )

    # Loading inventory data on which station measures which parameter, and since when.
    inventory = pd.read_csv(
        "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nime/ogd-nime_meta_datainventory.csv",
        sep=";",
        encoding="cp1252"
    )

    # Keeping only the stations that measure snow depth.
    snow_inventory = inventory[inventory["parameter_shortname"] == "hto000d0"]
    stations = stations[stations["station_abbr"].isin(snow_inventory["station_abbr"])]

    # Renaming and trimming stations to only include target columns.
    stations = stations.rename(columns={"station_abbr": "station_code",
                                       "station_coordinates_wgs84_lat": "lat",
                                       "station_coordinates_wgs84_lon": "lon",
                                       "station_height_masl": "altitude_m"
                                       })
    stations = stations[["station_code", "station_name", "lat", "lon", "altitude_m"]]

    return stations


@st.cache_data
def load_snow_depth(station_code, start_date, end_date):
    """
    Goal:       Timeseries of snow depth in cm for a certain station over a certain time interval.
    Input:      station_code ("DAV"), start_date and end_date ("2020-01-01", text or date)
    Website:    Not visible directly. Used for the labels, the model and the snow depth chart.
    Source:     MeteoSwiss Open Data (ch.meteoschweiz.ogd-nime).
    Note:       Downloads a file only when needed (historical once a year, recent once a day),
                saves it in SQLite and reads the asked dates from there. Missing days stay empty (NaN);
                start_date and end_date are inclusive
    Returns:    A table with station_code, date, snow_depth_cm.
    """

    # Making sure the database and its tables exist
    init_database()

    # The two MeteoSwiss files for this station (historical and recent (current year))
    code = station_code.lower()
    url_historical = f"https://data.geo.admin.ch/ch.meteoschweiz.ogd-nime/{code}/ogd-nime_{code}_d_historical.csv"
    url_recent = f"https://data.geo.admin.ch/ch.meteoschweiz.ogd-nime/{code}/ogd-nime_{code}_d_recent.csv"

    # Historical file: download and save only if needed, THEN record it in the logbook
    if needs_download("snow_historical", station_code):
        snow = download_snow_file(url_historical)
        save_snow_depth(snow)
        mark_downloaded("snow_historical", station_code)

    # Recent file: the same, with its own logbook entry
    if needs_download("snow_recent", station_code):
        snow = download_snow_file(url_recent)
        save_snow_depth(snow)
        mark_downloaded("snow_recent", station_code)

    return read_snow_depth(station_code, start_date, end_date)


@st.cache_data
def load_weather(lat, lon, start_date, end_date):
    """
    Goal:       Historical daily weather (temperature, snowfall) for one place.
    Input:      lat, lon (degrees), start_date and end_date ("2020-01-01", text or date, both included)
    Website:    Not visible. Extra input for the model, never the snow depth we predict.
    Source:     Open-Meteo Historical Weather API (ERA5 reanalysis), see WEATHER_URL.
    Note:       Open-Meteo answers for the nearest point of its weather grid, so the values are for a
                spot close to lat/lon, not exactly at it. The last few days can be empty (NaN),
                because the data lags a few days behind.
    Returns:    A table with date, temperature_c, snowfall_cm (one row per day).
    """
    # Code written by Claude Code (Anthropic) on 26.10.2026 at Nico's request, reviewed by Nico.

    # The API expects dates as text in the format yyyy-mm-dd
    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end = pd.to_datetime(end_date).strftime("%Y-%m-%d")

    # One request for the whole period; Swiss time zone, so a "day" means a Swiss day
    response = requests.get(
        WEATHER_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "daily": "temperature_2m_mean,snowfall_sum",
            "timezone": "Europe/Zurich",
        },
        timeout=60,
    )
    # Stop with a clear error if the API refuses (e.g. a wrong date), instead of continuing with nonsense
    response.raise_for_status()

    # The "daily" part holds equal-length lists (one entry per day), which become the table's columns
    weather = pd.DataFrame(response.json()["daily"])
    weather = weather.rename(columns={"time": "date",
                                      "temperature_2m_mean": "temperature_c",
                                      "snowfall_sum": "snowfall_cm"})
    weather["date"] = pd.to_datetime(weather["date"], format="%Y-%m-%d")

    return weather[["date", "temperature_c", "snowfall_cm"]]


@st.cache_data
def load_resorts():
    """
    Goal:       Swiss ski resorts with their slopes and lifts.
    Website:    The dots on the map and the facts in the resort card.
    Returns:    A table with resort_id, name, canton, lat, lon, base_altitude_m, summit_altitude_m,
                piste_km_total, piste_km_blue, piste_km_red, piste_km_black, lifts, source
    Note:       First version: collected with AI help from web sources and swisstopo (see GENAI_LOG.md).
                Version 2 will take piste km and lifts from OpenStreetMap.
    """

    # Reading the hand-made resort list
    resorts = pd.read_csv(RESORTS_PATH)

    # Keeping the columns in a fixed order
    resorts = resorts[["resort_id", "name", "canton", "lat", "lon", "base_altitude_m", "summit_altitude_m",
                       "piste_km_total", "piste_km_blue", "piste_km_red", "piste_km_black", "lifts", "source"]]

    return resorts


def link_stations_to_resorts(resorts, stations):
    """
    Goal:       Find the station that speaks for each resort, or none.
    Input:      The resorts table (load_resorts) and the stations table (load_stations).
    Website:    Resorts without a station are grey on the map and get no snow forecast.
    Note:       Uses the above linking rules (MAX_STATION_DISTANCE_KM, MAX_STATION_BELOW_BASE_M).
    Returns:    The resorts table plus station_code (empty if no good station is close)
    """

    resorts = resorts.copy()  # Copy to not change caller's table.
    station_codes = []

    for _, resort in resorts.iterrows():  # Iterating over all rows on the resorts table.
        candidates = stations.copy()  # Fresh copy of all stations for this resort.

        # Distance from this resort to every station (whole column at once).
        candidates["distance_km"] = measure_distance_km(resort["lat"], resort["lon"], candidates["lat"], candidates["lon"])

        # Determining min. allowed station altitude.
        lowest_allowed_m = resort["base_altitude_m"] - MAX_STATION_BELOW_BASE_M

        # Apply the rule: close enough, and not too far below the resort's base.
        candidates = candidates[(candidates["distance_km"] <= MAX_STATION_DISTANCE_KM) & (candidates["altitude_m"] >= lowest_allowed_m)]

        # The nearest suitable station wins; no suitable station means None.
        if candidates.empty:
            station_codes.append(None)
        else:
            nearest = candidates.sort_values("distance_km").iloc[0]  # Sorted near to far, so the first row is the nearest.
            station_codes.append(nearest["station_code"])  # Appends the station code of the nearest station to the station codes list.

    resorts["station_code"] = station_codes   # One station code (or None) per resort, as a new column.
    return resorts