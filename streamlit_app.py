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
import folium
from streamlit_folium import st_folium 

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

def travel_time_minutes(starting_place, resort):
    """
    Goal:     How long the trip by public transport takes from the user's city / village.
    Input:    starting_place (e.g. "Zurich"), one resort with transport_stop
    Website:  "Travel time" in the resort card, and the travel time filter.
    Returns:  minutes, or nothing if the place is unknown
    """
    # TODO [JAN | Task 8]
    # Hint:      transport.opendata.ch ("/v1/connections"). Only ask for a city or village, no
    #            street address. Dates far in the future may give no answer: use a normal weekday.
    # Done when: Zurich to Davos gives roughly 2 to 2.5 hours.
    destination_stop = resort["transport_stop"]

    # Ohne Zielhaltestelle können wir keine Verbindung abfragen.
    if pd.isna(destination_stop) or destination_stop == "":
        return None

    # Verbindung vom Startort zur Zielhaltestelle des Skigebiets abfragen.
    url = "https://transport.opendata.ch/v1/connections"
    parameter = {
        "from": starting_place,
        "to": destination_stop,
        "transportations[]": ["train", "bus"],
        "limit": 1,
    }


    # Anfrage senden; nach spätestens 15 Sekunden ohne Antwort abbrechen.
    try:
        antwort = requests.get(url, params=parameter, timeout=15)
        antwort.raise_for_status()

    except requests.exceptions.RequestException:
        # Reisezeit unbekannt, wenn die Abfrage fehlschlägt.
        return None

    # Die Antwort als Python-Daten einlesen.
    daten = antwort.json()
    verbindungen = daten["connections"]
    if not verbindungen:
        return None
    verbindung =  verbindungen [0]
    abfahrt = verbindung ["from"]["departureTimestamp"]
    ankunft = verbindung ["to"] ["arrivalTimestamp"]

    minuten = (ankunft - abfahrt)//60 
    return minuten
   



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


def filter_resorts(resorts, possible_cantons, starting_place, max_travelminutes):
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
    if possible_cantons:
        filtered = filtered[filtered["canton"].isin(possible_cantons)]
    
    if starting_place:
        passende_resorts =[]
        for index, resort in filtered.iterrows():
            minuten = travel_time_minutes(starting_place, resort) 
            if minuten is not None:
                 if minuten <= max_travelminutes: 
                    passende_resorts.append(index)
        filtered = filtered.loc[passende_resorts ]

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
    
    st.sidebar.subheader("Filters for your trip")

    # Starting location with which we calculate the travel time
    starting_place = st.sidebar.text_input(
        "Please enter your hometown with which we should calculate your travel time",
        placeholder="e.g. Zürich"
    )

    # Cantons in which the user possibly wants to travel
    possible_cantons = st.sidebar.multiselect(
        "Possible cantons for your trip",
        ["ZH", "BE", "LU", "UR", "GR", "VS", "TI", "GL"] # Possible anwsers to be adjusted later on
    )

    # Maximum time (in minutes) the user wants to travel from the given starting location to the disered destination
    max_travelminutes = st.sidebar.slider(
        "Maximum time (in minutes) you are willing to travel",
        min_value=30,
        max_value=300, # Should this be incresed later on? talk to jan
        step=15
    )

    # Timespan in which the user's holiday takes place
    travel_dates = 1 # Needs to be fully coded later on with help from Jan

    return starting_place, possible_cantons, max_travelminutes, travel_dates



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
    
    m = folium.Map(location=[46.8, 8.2], zoom_start=7) # This is the longitude and latitude of the center of Switzerland, so that the map is centered when opened

    for index, resort in resorts.iterrows():
    
        # Calculation of the radius of the circle, so that users can directly see the size of the resorts on the map
        map_radius = resort["piste_km_total"] / 10
    
        # This code ensures that the radius of the circle is not to big or to small, to ensure readability. Current problem: When zooming in our out the size of the circles does not change, so that the circles are to big when zooming in and to small when zooming out. This needs to be fixed later on.
        if map_radius > 15:
            map_radius = 15
        elif map_radius < 5:
            map_radius = 5 

        folium.CircleMarker(
        location=[resort["lat"], resort["lon"]], # Based on the latitude and longitude of the resorts (given by the data code), the circles are placed on the map
        radius = map_radius,
        popup=resort["name"],
        color="blue", # Colors should be adjusted later on based on the snow conditions, but for now they are all blue
        fill=True
        ).add_to(m)

    map_data = st_folium(m, width=None, height=600) # This code ensures the width and height of the map when displayed later on

    return None # Is this important?

    # Second problem: When the filter excludes some resorts, their circles are not displayed on the map. Is this as inteded or should those circles be red?



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


# The following section codes the user interface. Each Subsite of the overall website is divided by if page == xy
st.set_page_config(page_title="Snow-Proof", page_icon="🏔️", layout="wide")

st.sidebar.title("🏔️ Snow-Proof")
st.sidebar.caption("Your Swiss Snow Advisor")

st.sidebar.divider()

page = st.sidebar.radio(
    "Subsite-navigation",
    [
        "🏠 Home",
        "🎿 Holidayplanner",
        "🏡 Real Estate Advisor",
        "👥 About Us"
    ],)

# Here we code the landing page
if page == "🏠 Home":

    st.title("Welcome to Snow-Proof 🏔️")

    st.subheader(
        "Enjoy Skiing in Switzerland"
        "Optimize the planning of your trip."
    )

# Here we code the highly important holidayplaner as our main service
if page == "🎿 Holidayplanner":
    st.title("🎿 Holidayplanner")
    st.subheader(" This page helps you plan your Skiing trip based on the probability of great snow circumstances")
    starting_place, possible_cantons, max_travelminutes, travel_dates = show_sidebar()
    resorts = load_resorts()
    filtered_resorts = filter_resorts(resorts, possible_cantons, starting_place, max_travelminutes)
    st.write(f"{len(filtered_resorts)} of {len(resorts)} resorts match your filters.")
    st.dataframe(filtered_resorts) # Must be replaced by the map once the code is finished
    show_map(filtered_resorts, chances=None)

# Here we code the "Real Estate Advisor"
if page == "🏡 Real Estate Advisor":
    st.title("🏡 Alpine Real Estate Advisor")

# Here we code the "About Us" section
if page == "👥 About Us":
    st.title("👥 About Snow-Proof")

# =============================================================================
# SECTION 6: THE PAGE: PUTTING IT TOGETHER (Fabio, Task 12)
# Website part: the order of everything on the page. This is the code that actually runs.
# =============================================================================


# TODO [FABIO | Task 12]
# Goal:      Call the functions above in the right order, so the page works from top to bottom.
# Website:   The whole page: tabs "Explore", "About our model", "About us".
# Hint:      Order on the Explore tab: sidebar -> filter resorts -> map -> resort card -> snow check
#            -> ranked list. Connect each part as soon as its owner says it is done.
# Done when: a user can pick a resort, enter dates and see the chance, without errors.