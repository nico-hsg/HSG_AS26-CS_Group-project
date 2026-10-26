"""
Snow-Proof: getting the data (Section 1, Nico).

Loads MeteoSwiss stations and snow depth, and stores them in a SQLite database
(snowproof.db), so data is only downloaded when needed.
streamlit_app.py imports the functions from here.
"""

import streamlit as st
import pandas as pd
import sqlite3 as sql

# =============================================================================
# SECTION 1: GETTING THE DATA (Nico, Task 3 and 5)
# Website part: nothing visible directly. Everything else is built on this data.
# =============================================================================

DB_PATH = "snowproof.db"  # Database file, created in the project folder.

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
    Note:       Combines historical and recent (current year) data; Missing days stay empty (NaN); start_date and end_date are inclusive
    Returns:    A table with station_code, date, snow_depth_cm.
    """

    # Creating data source URL per station code (historical and recent (current year)).
    code = station_code.lower()
    url_historical = f"https://data.geo.admin.ch/ch.meteoschweiz.ogd-nime/{code}/ogd-nime_{code}_d_historical.csv"
    url_recent = f"https://data.geo.admin.ch/ch.meteoschweiz.ogd-nime/{code}/ogd-nime_{code}_d_recent.csv"

    # Loading station timeseries (historical and recent (current year)).
    historical = pd.read_csv(
        url_historical,
        sep=";",
        encoding="cp1252"
    )
    recent = pd.read_csv(
        url_recent,
        sep=";",
        encoding="cp1252"
    )

    # Stacking station timeseries, with continuous index.
    all_years = pd.concat([historical, recent], ignore_index=True)

    # Converting timeseries text dates into real dates, enabling correct filtering.
    all_years["date"] = pd.to_datetime(all_years["reference_timestamp"], format="%d.%m.%Y %H:%M")

    # Trimming timeseries to only include data from start date (inclusive) to end date (inclusive).
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    all_years = all_years[all_years["date"].between(start, end)]

    # Renaming and trimming timeseries to only include target columns, and refreshing index.
    all_years = all_years.rename(columns={"station_abbr": "station_code",
                                          "hto000d0": "snow_depth_cm"
                                          })
    all_years = all_years[["station_code", "date", "snow_depth_cm"]].reset_index(drop=True)

    return all_years


def load_weather(lat, lon, start_date, end_date):
    """
    Goal:     Historical daily weather (temperature, snowfall) for one place.
    Input:    lat, lon, start_date, end_date
    Website:  Not visible. Extra input for the model.
    Returns:  a table with date, temperature_c, snowfall_cm
    """
    # TODO [NICO | Task 3]
    # Hint:      Open-Meteo Historical API (archive-api.open-meteo.com), data from 1950.
    #            Use it only as model input, never as the snow depth we predict.
    # Done when: one month for Davos gives 30 or 31 rows.
    pass


def load_resorts():
    """
    Goal:     All Swiss ski resorts with their slopes and lifts.
    Website:  The dots on the map and the facts in the resort card.
    Returns:  a table with resort_id, name, canton, lat, lon, base_altitude_m, summit_altitude_m,
              piste_km_blue, piste_km_red, piste_km_black, lifts
    """
    # TODO [NICO | Task 5]
    # Hint:      OpenStreetMap via the Overpass API (overpass.osm.ch). Ski areas are
    #            "landuse=winter_sports", slopes "piste:type=downhill", lifts "aerialway".
    #            Piste km comes from the length of the slope lines, not from counting slopes.
    #            A hand-made list of 20 to 30 resorts is a fine first version.
    # Done when: Zermatt, Davos and Laax are in the table with piste km and lifts.
    pass