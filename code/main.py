import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))

import pandas as pd

from files import clean_output, make_folder, write_notes
from graphs import make_plots
from models import choose_features, compare_features, train_models


SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(SCRIPT_FOLDER)
DATA_FILE = os.path.join(PROJECT_FOLDER, "Network Attack", "network_attack.xlsx")
OUT_FOLDER = os.path.join(PROJECT_FOLDER, "output")
GRAPH_FOLDER = os.path.join(OUT_FOLDER, "graphs")
TARGET = "label"


def load_data():
    data = pd.read_excel(DATA_FILE)
    data = data.dropna(subset=[TARGET])
    data[TARGET] = data[TARGET].astype(int)
    return data


def main():
    make_folder(OUT_FOLDER)
    make_folder(GRAPH_FOLDER)
    data = load_data()

    clean_output(OUT_FOLDER)
    make_plots(data, GRAPH_FOLDER, TARGET)

    features = choose_features(data, OUT_FOLDER, TARGET)
    results = train_models(data, features, OUT_FOLDER, TARGET)
    comparison = compare_features(data, features, OUT_FOLDER, TARGET)

    write_notes(data, features, results, comparison, OUT_FOLDER, TARGET)

    print(results.to_string(index=False))
    print(f"\nDone. Files are in {OUT_FOLDER}")


if __name__ == "__main__":
    main()
