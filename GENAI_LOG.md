# GenAI log

We log every use of generative AI here: what we asked, what we used, and what we changed ourselves.
This is part of our grade (requirement 7) and of the video.

## Starting point

We used Claude Code to generate the starting structure of the project: the sections of `streamlit_app.py`, the
function names, docstrings and TODO placeholders, the documentation templates, and a clickable UI prototype with mock data (branch
`ui-prototype`) as a visual reference. All logic behind the TODOs is written by us.

## Log

| Date | Person | Tool | What we asked | What we used | What we changed ourselves |
|---|---|---|---|---|---|
| 22.10. to 24.10.2026 | Team | Claude Code | Project structure, UI prototype, then simplified to one file with placeholders only | one-file skeleton (branch `skeleton-one-page`), docs templates, prototype as visual reference | |
| 25.10.2026 | Nico | Claude Code | Explain the MeteoSwiss data and the pandas tools for load_stations, and review my code | Explanations and review only | Wrote all of load_stations myself |
| 25.10.2026 | Nico | Claude Code | Explain the MeteoSwiss daily files and the pandas tools for load_snow_depth, and review my code | Explanations and review only | Wrote all of load_snow_depth myself |
| 26.10.2026 | Nico | Claude Code | Explain how to set up the SQLite database and .gitignore, and review my code | Explanations and review only | Wrote init_database myself |
| 26.10.2026 | Nico | Claude Code | Explain how to save and read pandas tables with SQLite, and review my code | Explanations and review only | Wrote save_snow_depth and read_snow_depth myself |
| 26.10.2026 | Nico | Claude Code | Explain the download logbook (update_table) and give fill-in templates for mark_downloaded, needs_download and load_snow_depth; review my code | Explanations, templates with blanks, review | Filled in the logic and wrote download_snow_file myself |
| 26.10.2026 | Nico | Claude Code | Collect the data for `resorts.csv` (20 resorts): coordinates from the swisstopo search API, summit heights from the swisstopo height API where the top lift reaches a named peak (7 resorts), other piste km, lifts and altitudes from web searches | The whole `resorts.csv`, as a first version until we compute piste km and lifts from OpenStreetMap; each row cites its sources | Chose the approach (hand-made list first, OpenStreetMap later) and which resorts to include |
| 26.10.2026 | Nico | Claude Code | Explain how to match stations to resorts (rule options, distance formula, a column-based approach segment by segment); review my code | Explanations and review | Chose the rule, wrote measure_distance_km and link_stations_to_resorts myself |
| 26.10.2026 | Nico | Claude Code | Write load_weather (Open-Meteo Historical Weather API) because of a time budget | The whole function and the WEATHER_URL constant, cited in the source code | Reviewed and tested it |
| | | | | | |

## Reflection (for the video)

- What did GenAI help us with?
- Where was it wrong or not helpful?
- What did we learn by writing the logic ourselves?
