"""Fetch the open validation games into data/external/ (gitignored):

    python validation/download.py              # Metrica's three public sample games, ~185 MB

These are *other* games, which RULES.md allows for training and validation in any amount. They are
public, so the organisers can fetch exactly the same files with this script.
"""

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL = os.path.join(ROOT, "data", "external")

METRICA_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
METRICA_FILES = (
    "Sample_Game_1/Sample_Game_1_RawEventsData.csv",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
    "Sample_Game_2/Sample_Game_2_RawEventsData.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
    "Sample_Game_3/Sample_Game_3_events.json",
    "Sample_Game_3/Sample_Game_3_metadata.xml",
    "Sample_Game_3/Sample_Game_3_tracking.txt",
)


def fetch(url, path):
    if os.path.isfile(path):
        print("have   ", os.path.relpath(path, ROOT))
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("fetch  ", url)
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)


def metrica():
    for name in METRICA_FILES:
        fetch(METRICA_URL + name, os.path.join(EXTERNAL, "metrica", name))


if __name__ == "__main__":
    metrica()
    sys.exit(0)
