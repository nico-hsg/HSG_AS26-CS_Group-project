"""
Snow-Proof: where will the snow be good for your ski holiday?

The app shows Swiss ski resorts on a map. The user clicks a resort, sees its slopes, and checks
the chance of a "good ski week" for their holiday dates. A good ski week means at least 50 cm of
measured snow depth on all 7 days at the station next to the resort. The chance comes from our
own machine learning model, and we always compare it with the simple historical average.

Run it:   streamlit run streamlit_app.py

How this file is organised (top to bottom, in the order the app needs it):

  Section 0  Settings                         Fabio
  Section 1  Getting the data                 Nico
  Section 2  Good-week labels                 Nico + Julian
  Section 3  The machine learning model       Julian (+ Jan)
  Section 4  Location, travel, filters        Jan
  Section 5  What the user sees               Noah (+ Jan); the two About tabs: Fabio
  Section 6  The page: putting it together    Fabio

Find your work: search this file for "TODO [YOURNAME" (for example "TODO [NICO").
Each placeholder tells you the goal, which part of the website it builds, a short hint,
and when it is done. HOW you write it is up to you.

Rule for working together: only edit your own section. Git can combine changes from
different sections automatically, but not when two people change the same lines.
"""

import streamlit as st
import pandas as pd
import requests
import sqlite3 as sql

# Add more imports here when you need them, for example:
#   folium and streamlit_folium (map), plotly (charts), sklearn (model)


# =============================================================================
# SECTION 0: SETTINGS (Fabio)
# The rules of our project in one place. Change a number here, and the whole app follows.
# =============================================================================

GOOD_SNOW_CM = 50                 # minimum snow depth on every day of the week
GOOD_WEEK_DAYS = 7                # a ski week has 7 days
SEASON_MONTHS = [12, 1, 2, 3, 4]  # December to April


# =============================================================================
# SECTION 1: GETTING THE DATA (Nico, Task 3 and 5)
# Website part: nothing visible directly. Everything else is built on this data.
# =============================================================================

DB_PATH = "snowproof.db" # Database file, created in the project folder.

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


# =============================================================================
# SECTION 2: GOOD-WEEK LABELS (Nico + Julian, Task 4)
# Website part: nothing visible directly. This is the "answer" the model learns from.
# =============================================================================

def link_stations_to_resorts(resorts, stations):
    """
    Goal:     Find the station that speaks for each resort, or none.
    Input:    the resorts table and the stations table
    Website:  Resorts without a station are grey on the map and get no snow forecast.
    Returns:  the resorts table plus station_code (empty if no good station is close)
    """
    # TODO [NICO | Task 4]
    # Hint:      Close by (e.g. within 15 km) and at a similar altitude. Write down your rule,
    #            you will explain it in the video.
    # Done when: Davos gets DAV, and some low or remote resorts get no station.
    pass


def label_good_weeks(snow_depth):
    """
    Goal:     For every week: was it a good ski week? 1 = yes, 0 = no, empty = unknown.
    Input:    the snow depth table from load_snow_depth()
    Website:  Not visible directly. The model learns from these labels.
    Returns:  a table with station_code, week_start, year, good_week
    """
    # TODO [NICO+JULIAN | Task 4]
    # Hint:      Good = at least GOOD_SNOW_CM on all GOOD_WEEK_DAYS days. One day below -> 0.
    #            A missing day (and no day below) -> unknown. Only SEASON_MONTHS count.
    # Done when: you checked a few weeks by hand and they match the rule.
    pass


# =============================================================================
# SECTION 3: THE MACHINE LEARNING MODEL (Julian, Task 6 and 7; Jan helps)
# Website part: the chance in the snow check, and the "About our model" tab.
# =============================================================================

def historical_average(labels, station_code, week_of_year):
    """
    Goal:     The simple guess our model must beat: how often was this week good in the past?
    Input:    the labels, one station, one week of the year
    Website:  "Historical average for this week" in the snow check.
    Returns:  a number between 0 and 1
    """
    # TODO [JULIAN | Task 6]
    # Hint:      The share of good weeks in all earlier years. Skip unknown weeks.
    # Done when: Davos in mid-February gives a high value, a low station in December a low one.
    pass


def train_model(labels, stations):
    """
    Goal:     Train our own model that predicts the chance of a good week.
    Input:    the labels and the stations (for altitude)
    Website:  Not visible directly. Powers the chance in the snow check.
    Returns:  the trained model
    """
    # TODO [JULIAN+JAN | Task 6]
    # Hint:      Only use inputs you know BEFORE the holiday: altitude, week of the season, year,
    #            past share of good weeks. Never the weather of the holiday week itself.
    #            Try two scikit-learn models (e.g. logistic regression, random forest), keep the better.
    #            Streamlit's @st.cache_resource trains it only once.
    # Done when: the model gives a chance between 0 and 1 for any station and week.
    pass


def test_model(labels, stations):
    """
    Goal:     Show that the model works: train on the past, test on later years, compare with
              the historical average.
    Input:    the labels and the stations
    Website:  The results table and chart in the "About our model" tab.
    Returns:  a table with one row per round: train years, test years, model score, average score
    """
    # TODO [JULIAN | Task 7]
    # Hint:      Walk-forward: train 1950-1969 test 1970-1979, train 1950-1979 test 1980-1989, ...
    #            Final check: train 1950-2014, test 2015-2025, used only once.
    #            Scores: accuracy and Brier score. Change the years if the stations start later.
    # Done when: you can say in one sentence whether the model beats the historical average.
    pass


def predict_good_week(model, resort, start_date, end_date):
    """
    Goal:     The chance of a good ski week for one resort and the user's dates.
    Input:    the trained model, one resort, the holiday dates
    Website:  The big number "68% chance of a good ski week" in the snow check.
    Returns:  a number between 0 and 1 (or nothing if the resort has no station)
    """
    # TODO [JULIAN | Task 7]
    # Hint:      Build the same inputs as in train_model for these dates, then ask the model.
    # Done when: a high resort in February gets a higher chance than a low resort in December.
    pass


# =============================================================================
# SECTION 4: LOCATION, TRAVEL AND FILTERS (Jan, Task 8 and 9)
# Website part: the filters in the sidebar and the ranked list below the map.
# =============================================================================

def travel_time_minutes(place, resort):
    """
    Goal:     How long the trip by public transport takes from the user's city / village.
    Input:    place (e.g. "Zurich"), one resort
    Website:  "Travel time" in the resort card, and the travel time filter.
    Returns:  minutes, or nothing if the place is unknown
    """
    # TODO [JAN | Task 8]
    # Hint:      transport.opendata.ch ("/v1/connections"). Only ask for a city or village, no
    #            street address. Dates far in the future may give no answer: use a normal weekday.
    # Done when: Zurich to Davos gives roughly 2 to 2.5 hours.
    pass


def school_holidays(canton):
    """
    Goal:     School holiday dates of a canton, to suggest holiday dates.
    Input:    canton code, e.g. "ZH"
    Website:  The "use school holidays" helper in the sidebar.
    Returns:  a list of (start_date, end_date)
    """
    # TODO [JAN | Task 8]
    # Hint:      OpenHolidays API ("/SchoolHolidays"). Sportferien are often missing, so the user
    #            must always be able to pick dates by hand.
    # Done when: Zurich shows its Christmas holidays.
    pass


def filter_resorts(resorts, cantons, place, max_minutes):
    """
    Goal:     Keep only the resorts that match the user's filters. All filters are optional.
    Input:    the resorts, the chosen cantons, the city / village, the max travel time
    Website:  Which dots stay on the map, and "14 of 23 resorts match".
    Returns:  the filtered resorts table
    """
    # TODO [JAN | Task 8]
    # Hint:      With no filter set, return all resorts.
    # Done when: choosing "VS" leaves only Valais resorts.
    pass


def rank_resorts(resorts, chances):
    """
    Goal:     Sort the resorts from best to worst chance of a good week.
    Input:    the resorts and their chances for the user's dates
    Website:  The ranked list below the map (only when holiday dates are set).
    Returns:  the resorts table sorted by chance, best first
    """
    # TODO [JAN | Task 9]
    # Hint:      Leave out resorts without a station.
    # Done when: the first row has the highest chance.
    pass


# =============================================================================
# SECTION 5: WHAT THE USER SEES (Noah, Task 10 and 11; the two About tabs: Fabio)
# Website part: everything on the screen. Look at the ui-prototype branch for the look.
# =============================================================================

def show_sidebar():
    """
    Goal:     The optional filters on the left side.
    Website:  Sidebar: city / village, canton, max travel time, holiday dates, school holiday helper.
    Returns:  what the user chose (e.g. a dictionary)
    """
    # TODO [NOAH+JAN | Task 10]
    # Hint:      st.sidebar with text input, multiselect, slider and date input.
    # Done when: changing a filter changes what the map shows.
    pass


def show_map(resorts, chances):
    """
    Goal:     The map of Switzerland with one dot per resort.
    Input:    the (filtered) resorts, and their chances if holiday dates are set
    Website:  The big map at the top, full width.
    Returns:  the resort the user clicked, or nothing
    """
    # TODO [NOAH | Task 11]
    # Hint:      folium with the swisstopo map, and streamlit_folium to show it and read the click.
    #            Bigger dot = more piste km, grey = no station, traffic light colors with dates.
    # Done when: clicking a dot opens that resort's card below the map.
    pass


def show_resort_card(resort):
    """
    Goal:     The facts of the clicked resort.
    Input:    one resort
    Website:  The card below the map: piste km, lifts, altitudes, slopes by color, station,
              travel time, and the button "Check the snow for your holiday".
    Returns:  nothing
    """
    # TODO [NOAH | Task 11]
    # Hint:      st.metric for the numbers, a plotly bar for blue / red / black km.
    #            st.session_state remembers the clicked resort between clicks.
    # Done when: every resort shows its card, and resorts without a station show a warning.
    pass


def show_snow_check(resort, chance, average, snow_depth):
    """
    Goal:     The answer to the user's question for their dates.
    Input:    the resort, the chance, the historical average, the snow depth per day
    Website:  The snow check: date input, a 0 to 100% bar with the chance and the historical
              average, the expected snow depth chart with the 50 cm line, and "About this forecast".
    Returns:  nothing
    """
    # TODO [NOAH | Task 11]
    # Hint:      Only allow dates from December to April, and say so in a friendly way.
    # Done when: it looks like the snow check on the ui-prototype branch.
    pass


def show_about_model(results):
    """
    Goal:     Explain our model to users and graders.
    Input:    the results table from test_model()
    Website:  The "About our model" tab: data sources, what a good week is, inputs, results chart.
    Returns:  nothing
    """
    # TODO [FABIO+JULIAN | Task 10]
    # Hint:      Julian tells you what to explain. Short texts and one chart: model versus historical average per round.
    # Done when: someone outside the course understands what we predict and how well.
    pass


def show_about_us():
    """
    Goal:     Who we are and why we built Snow-Proof.
    Website:  The "About us" tab.
    Returns:  nothing
    """
    # TODO [FABIO | Task 10]
    # Hint:      A few sentences and one line per team member.
    # Done when: everyone checked their own line.
    pass


# =============================================================================
# SECTION 6: THE PAGE: PUTTING IT TOGETHER (Fabio, Task 12)
# Website part: the order of everything on the page. This is the code that actually runs.
# =============================================================================

st.set_page_config(page_title="Snow-Proof", page_icon="🏔️", layout="wide")
st.title("🏔️ Snow-Proof")
st.write("Where will the snow be good for your ski holiday?")
st.info("Under construction. The sections above are being built by the team.")

# TODO [FABIO | Task 12]
# Goal:      Call the functions above in the right order, so the page works from top to bottom.
# Website:   The whole page: tabs "Explore", "About our model", "About us".
# Hint:      Order on the Explore tab: sidebar -> filter resorts -> map -> resort card -> snow check
#            -> ranked list. Connect each part as soon as its owner says it is done.
# Done when: a user can pick a resort, enter dates and see the chance, without errors.
