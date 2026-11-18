
from pathlib import Path

target_path = Path("streamlit_app.py")
source_path = Path("julian_ml_reference.py")

target = target_path.read_text(encoding="utf-8")
source = source_path.read_text(encoding="utf-8")

section_2 = "# SECTION 2: GOOD-WEEK LABELS"
section_4 = "# SECTION 4: LOCATION, TRAVEL AND FILTERS"

def find_section_start(text, heading):
    heading_position = text.index(heading)
    return text.rfind(
        "# =============================================================================",
        0,
        heading_position
    )

source_start = find_section_start(source, section_2)
source_end = find_section_start(source, section_4)

target_start = find_section_start(target, section_2)
target_end = find_section_start(target, section_4)

# Copy only Sections 2 and 3 from Julian's version
new_sections = source[source_start:source_end]

updated = (
    target[:target_start]
    + new_sections
    + target[target_end:]
)

# Add ML imports without removing existing imports
ml_imports = [
    "from sklearn.ensemble import RandomForestClassifier",
    "from sklearn.compose import ColumnTransformer",
    "from sklearn.preprocessing import OneHotEncoder",
    "from sklearn.pipeline import Pipeline",
]

missing_imports = [
    line for line in ml_imports
    if line not in updated.splitlines()
]

if missing_imports:
    anchor = "import pandas as pd\n"
    assert updated.count(anchor) == 1, "Import position unclear"
    updated = updated.replace(
        anchor,
        anchor + "\n".join(missing_imports) + "\n",
        1
    )

# Ensure everything after Section 4 remains unchanged
assert (
    updated[find_section_start(updated, section_4):]
    == target[target_end:]
), "Section 4 or later sections changed!"

# Save the integration
target_path.write_text(updated, encoding="utf-8")

print("ML integration completed.")
print("Sections 4, 5 and 6 preserved.")
