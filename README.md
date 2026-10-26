# Snow-Proof

Snow-Proof is a Streamlit web app that shows Swiss ski resorts on a map. Click a resort to see its slopes and key
facts, then check the chance of a **good ski week** for your holiday dates: at least 50 cm of measured snow depth on all
7 days at the station next to the resort. The chance comes from **our own machine learning model**, trained on measured
snow depth (MeteoSwiss) and historical weather (Open-Meteo), and it is always compared with the simple historical average.

**Status: skeleton.** The app lives in `streamlit_app.py`, except Section 1 (data), which is in `data.py`. Every part has a placeholder with its goal,
the part of the website it builds, a short hint and a "Done when" check. We write the code ourselves. The planned look is on the `Prototype` branch.

## What is in this repository

```
streamlit_app.py     everything except Section 1, one section per person
data.py              Section 1: loading data from the APIs, and the SQLite database
requirements.txt     the Python packages the app needs
README.md            this file
CONTRIBUTIONS.md     who did what
GENAI_LOG.md         how we used AI tools
.gitignore           files git should not upload (temporary Python files, virtual environment, SQLite database)
```

## Sections in streamlit_app.py and data.py

| Section | Owner | Part of the website |
|---|---|---|
| 0 Settings | Fabio | the rules: 50 cm, 7 days, December to April |
| 1 Getting the data in data.py | Nico | nothing visible, everything is built on it |
| 2 Good-week labels | Nico + Julian | nothing visible, the "answer" the model learns from |
| 3 Machine learning model | Julian (+ Jan) | the chance in the snow check, the "About our model" tab |
| 4 Location, travel, filters | Jan | sidebar filters, travel time, ranked list |
| 5 What the user sees | Noah (+ Jan) | sidebar, map, resort card, snow check |
| 5 the two "About" tabs | Fabio (+ Julian) | "About our model" and "About us" tabs |
| 6 Putting it together | Fabio | the order of everything on the page |

## Run it on your computer

```bash
git clone https://github.com/nico-hsg/HSG_AS26-CS_Group-project.git
cd HSG_AS26-CS_Group-project
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate; if activation is blocked, write .venv\Scripts\python -m … in front of the commands instead
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Find your tasks

Search `streamlit_app.py` and `data.py` for `TODO [YOURNAME` (for example `TODO [NICO`). Shared tasks look like `TODO [NICO+JULIAN`,
so the same search finds them. Each placeholder looks like this:

```python
# TODO [NICO | Task 3]
# Hint:      where the data comes from, or which tool helps
# Done when: a simple check that tells you it works
pass
```

Replace `pass` with your code when you start. The task numbers come from our Contribution Matrix:

| Person | Tasks |
|---|---|
| Fabio | 1 Project management, 2 Technical foundation, 12 Integration and testing, 13 Final review, plus the two "About" tabs (from task 10) |
| Nico | 3 Snow and weather data, 4 Data preparation, 5 Ski resort information |
| Julian | 6 Model training, 7 Model evaluation and prediction |
| Jan | 8 Location, travel and holiday filters, 9 Resort ranking |
| Noah | 10 User interface and navigation, 11 Map, charts and resort details |
| Everyone | Video, contribution documentation, GenAI reflection |

## Working together with git

1. Before you start: `git pull` to get the newest version.
2. **Only edit your own section.** Git combines changes from different sections automatically.
3. Make small commits with clear messages, e.g. `git commit -m "Load snow depth for one station"`.
4. Push and open a pull request into `main`. One teammate looks at it before merging.
5. If you need something changed in another section, ask its owner.

## Plan until 10.12.2026

| Until | Milestone |
|---|---|
| 26.10. | Everyone can run the app on their computer |
| 08.11. | Data loads (Nico), resort list and map (Noah), filters (Jan) |
| 15.11. | Good-week labels checked (Nico + Julian), resort card and snow check layout (Noah) |
| 26.11. | Model trained and tested against the historical average (Julian + Jan) |
| 03.12. | Everything connected and working (Fabio) |
| 09.12. | Video recorded, CONTRIBUTIONS.md and GENAI_LOG.md complete (everyone) |
| 10.12. | Submission |

Important order: the labels must be ready before the model can be trained. The page layout can be built at any time.

## Grading map

| # | Course requirement | Where we cover it |
|---|---|---|
| 1 | Problem clearly stated | top of `streamlit_app.py`, this README, "About us" tab, video |
| 2 | Data via API and/or database | Section 1 (MeteoSwiss, Open-Meteo, OpenStreetMap) in `data.py` with SQLite and Section 4 (transport, holidays) |
| 3 | Data visualization | Section 5: map, chance bar, snow depth chart, model results chart |
| 4 | User interaction | Section 5: filters, map clicks, date input |
| 5 | Own trained ML model | Section 3, results in the "About our model" tab |
| 6 | Code documented with comments | every function has a docstring and comments |
| 7 | Team contributions and GenAI use documented, shown in the video | `CONTRIBUTIONS.md`, `GENAI_LOG.md`, video |
| 8 | 4-minute demo video with our own (not AI-generated) voice-over | recorded by everyone |

## Data sources

MeteoSwiss Open Data, Open-Meteo (ERA5), © OpenStreetMap contributors, © swisstopo, transport.opendata.ch,
OpenHolidays API.
