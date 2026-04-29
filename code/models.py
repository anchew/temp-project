import os

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


SAMPLE_SIZE = 50000


def choose_features(data, out_folder, target):
    numbers = data.select_dtypes(include="number").drop(columns=["id"], errors="ignore")
    corr = numbers.corr(numeric_only=True)[target].abs().sort_values(ascending=False)

    features = corr.drop(labels=[target], errors="ignore").head(20).index.tolist()
    for name in ["proto", "service", "state"]:
        if name in data.columns:
            features.append(name)

    return features


def fix_missing_values(data):
    data = data.copy()

    for name in data.columns:
        if is_numeric_dtype(data[name]):
            data[name] = data[name].fillna(data[name].median())
        else:
            data[name] = data[name].fillna(data[name].mode()[0])

    return data


def make_model_data(data, features, target):
    if len(data) > SAMPLE_SIZE:
        data, _ = train_test_split(
            data,
            train_size=SAMPLE_SIZE,
            random_state=42,
            stratify=data[target],
        )

    x = data[features]
    y = data[target]

    x = fix_missing_values(x)
    x = pd.get_dummies(x, drop_first=True)

    return train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)


def test_one_model(data, features, target, model):
    x_train, x_test, y_train, y_test = make_model_data(data, features, target)
    model.fit(x_train, y_train)
    guesses = model.predict(x_test)

    return {
        "accuracy": accuracy_score(y_test, guesses),
        "f1": f1_score(y_test, guesses),
        "report": classification_report(y_test, guesses),
    }


def train_models(data, features, out_folder, target):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=60, max_depth=14, random_state=42, n_jobs=-1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=60, max_depth=14, random_state=42, n_jobs=-1
        ),
    }

    results = []
    reports = []

    for name, model in models.items():
        score = test_one_model(data, features, target, model)

        results.append(
            {
                "model": name,
                "accuracy": score["accuracy"],
                "f1": score["f1"],
            }
        )
        reports.append(name)
        reports.append(score["report"])

    results = pd.DataFrame(results).sort_values("f1", ascending=False)
    results.to_csv(os.path.join(out_folder, "model_results.csv"), index=False)
    save_text(out_folder, "model_reports.txt", reports)

    return results


def compare_features(data, selected_features, out_folder, target):
    all_features = [
        name
        for name in data.columns
        if name not in [target, "attack_cat", "id"]
    ]
    model = RandomForestClassifier(
        n_estimators=60, max_depth=14, random_state=42, n_jobs=-1
    )

    selected_score = test_one_model(data, selected_features, target, model)
    all_score = test_one_model(data, all_features, target, model)

    rows = [
        {
            "feature_set": "selected_features",
            "feature_count": len(selected_features),
            "accuracy": selected_score["accuracy"],
            "f1": selected_score["f1"],
        },
        {
            "feature_set": "all_features_except_attack_cat",
            "feature_count": len(all_features),
            "accuracy": all_score["accuracy"],
            "f1": all_score["f1"],
        },
    ]

    comparison = pd.DataFrame(rows)
    return comparison


def save_text(out_folder, file_name, lines):
    with open(os.path.join(out_folder, file_name), "w") as file:
        file.write("\n".join(lines))
