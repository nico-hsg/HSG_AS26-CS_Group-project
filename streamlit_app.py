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
from data import load_stations, load_snow_depth, load_weather, load_resorts, link_stations_to_resorts
from sklearn.ensemble import RandomForestClassifier

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
# SECTION 1: GETTING THE DATA IN data.py (Nico, Task 3 and 5)
# Lives in data.py. The functions are imported at the top of this file.
# =============================================================================


# =============================================================================
# SECTION 2: GOOD-WEEK LABELS (Nico + Julian, Task 4)
# Website part: nothing visible directly. This is the "answer" the model learns from.
# =============================================================================


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


# =============================================================================
# SECTION 3: THE MACHINE LEARNING MODEL (Julian, Task 6 and 7; Jan helps)
# Website part: the chance in the snow check, and the "About our model" tab.
# =============================================================================

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




@st.cache_resource
def train_model(labels, stations):
    """
    Train a Random Forest to predict good ski weeks.
    A good week has an average snow depth of at least
    50 cm over seven consecutive days.
    """

    # Combine ski week labels with station altitude
    data = labels.merge(
        stations[["station_code", "altitude_m"]],
        on="station_code",
        how="inner"
    )

    # Remove missing observations
    data = data.dropna(
        subset=["good_week", "altitude_m", "week_start"]
    ).copy()

    # Calculate the last day of each ski week
    data["week_end"] = data["week_start"] + pd.Timedelta(days=6)

    # Keep 2024 completely separate for final testing
    data = data[
        data["week_end"] < pd.Timestamp("2024-01-01")
    ].copy()

    # Extract the date features
    data["month"] = data["week_start"].dt.month
    data["day"] = data["week_start"].dt.day

    # Define features and labels
    features = ["altitude_m", "month", "day"]
    X = data[features]
    y = data["good_week"].astype(int)

    # Build the same Random Forest as in our tests
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42
    )

    # Train the model
    model.fit(X, y)

    # Remember which stations were used for training
    model.snowproof_station_codes = set(data["station_code"])

    # Remember the altitude of every trained station
    model.snowproof_station_altitudes = dict(
        zip(data["station_code"], data["altitude_m"])
    )

    return model



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
    start = "Zürich HB"
    ziel = "Davos Platz"

    # Testabfrage: zunächst nur Zürich HB nach Davos Platz.
    url = "https://transport.opendata.ch/v1/connections"
    parameter = {
        "from": start,
        "to": ziel,
        "transportations[]": ["train", "bus"],
        "limit": 1,
    }

    # Anfrage senden; nach spätestens 15 Sekunden ohne Antwort abbrechen.
    antwort = requests.get(url, params=parameter, timeout=15)
    antwort.raise_for_status()

    # Die Antwort als Python-Daten einlesen.
    daten = antwort.json()

    # TODO: Aus daten die Reisezeit in Minuten berechnen.



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
    filtered = resorts.copy()

    # Nur Skigebiete aus den ausgewählten Kantonen behalten.
    if cantons:
        filtered = filtered[filtered["canton"].isin(cantons)]

    # TODO: Den Reisezeitfilter ergänze ich im nächsten Schritt.
    return filtered


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
    # chances enthält die Vorhersagen für eine Woche (Montag bis Sonntag).
    columns = ["resort_id", "week_start", "week_end", "probability"]
    chances_table = pd.DataFrame(chances, columns=columns)

    # 1. Skigebiete und Vorhersagen über ihre ID verbinden.
    ranked = resorts.merge(chances_table, on="resort_id")

    # 2. Gebiete ohne Vorhersage oder zugeordnete Messstation weglassen.
    ranked = ranked.dropna(subset=["probability"])
    if "station_code" in ranked.columns:
        ranked = ranked.dropna(subset=["station_code"])

    # 3. Nach Wahrscheinlichkeit sortieren: höchste zuerst.
    ranked = ranked.sort_values("probability", ascending=False)

    # 4. Zeilennummern neu setzen und die Rangliste zurückgeben.
    ranked = ranked.reset_index(drop=True)
    return ranked


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