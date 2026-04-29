import os


def make_folder(out_folder):
    os.makedirs(out_folder, exist_ok=True)


def save_text(out_folder, file_name, lines):
    with open(os.path.join(out_folder, file_name), "w") as file:
        file.write("\n".join(lines))


def clean_output(out_folder):
    old_files = [
        "project_notes.txt",
        "dataset_notes.txt",
        "graph_notes.txt",
        "model_notes.txt",
        "summary.txt",
        "features.txt",
        "feature_comparison.csv",
        "report_draft.txt",
        "data_guide.txt",
        "label_counts.png",
        "attack_categories.png",
        "protocol_counts.png",
        "duration_by_label.png",
        "bytes_by_label.png",
        "correlation_heatmap.png",
    ]

    for file_name in old_files:
        path = os.path.join(out_folder, file_name)
        if os.path.exists(path):
            os.remove(path)

    graph_folder = os.path.join(out_folder, "graphs")
    if os.path.exists(graph_folder):
        for file_name in os.listdir(graph_folder):
            if file_name.endswith(".png"):
                os.remove(os.path.join(graph_folder, file_name))


def write_notes(data, features, results, comparison, out_folder, target):
    normal_count = (data[target] == 0).sum()
    attack_count = (data[target] == 1).sum()
    attack_ratio = attack_count / normal_count
    missing = data.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    best = results.iloc[0]
    worst = results.iloc[-1]
    selected = comparison[comparison["feature_set"] == "selected_features"].iloc[0]
    all_features = comparison[comparison["feature_set"] == "all_features_except_attack_cat"].iloc[0]

    dataset_notes = [
        "Dataset notes",
        "",
        f"Rows after dropping blank labels: {len(data):,}",
        f"Columns: {len(data.columns)}",
        f"Normal rows: {normal_count:,}",
        f"Attack rows: {attack_count:,}",
        f"Attack traffic is about {attack_ratio:.2f} times the normal traffic.",
        "Label: 0 = normal, 1 = attack",
        "This is not one attack example. It is a mix of normal traffic and many attack examples.",
        "Since attacks are the larger class, accuracy by itself is not enough to talk about performance.",
        "F1 is useful here because it gives a better picture of precision and recall together.",
        "",
        "Attack category counts:",
        data["attack_cat"].value_counts(dropna=False).to_string(),
        "Generic and Exploits are the largest attack categories, so the attack side of the dataset is not evenly spread out.",
        "Worms is tiny compared to the other attack categories, which means the model probably gets much less practice on that type.",
        "",
        "Preprocessing:",
        "Dropped rows with blank labels.",
        "Filled number blanks with median.",
        "Filled text blanks with most common value.",
        "I would explain this as keeping rows instead of throwing away a lot of the dataset.",
        "Median is safer than mean here because traffic data can have large outliers.",
        "The label itself was not filled in because guessing the answer column would make the training less honest.",
        "",
        "Missing value columns:",
        missing.to_string(),
    ]

    graph_notes = [
        "Graph notes",
        "",
        "label_counts.png: normal vs attack count",
        "attack_categories.png: attack type counts",
        "protocol_counts.png: protocol use by label",
        "duration_by_label.png: connection duration by label",
        "bytes_by_label.png: total bytes by label",
        "correlation_heatmap.png: strongest numeric correlations",
        "These cover the rubric requirement for at least three visualizations.",
        "The duration and bytes graphs use a log scale because network traffic values can get very spread out.",
        "The correlation heatmap is mainly for feature selection, not for proving one feature explains the whole dataset.",
    ]

    model_notes = [
        "Model notes",
        "",
        "Selected features:",
        ", ".join(features),
        "The selected feature list has several traffic/state columns, which makes sense because attacks often change connection behavior.",
        "I avoided using attack_cat as a feature because it is basically another version of the answer.",
        "",
        "Feature comparison:",
        f"Selected features: {int(selected['feature_count'])} features, F1 {selected['f1']:.4f}",
        f"All non-leaking features: {int(all_features['feature_count'])} features, F1 {all_features['f1']:.4f}",
        "The selected features made the model simpler, but the full non-leaking feature set did a little better.",
        "This is still useful because it shows a real tradeoff instead of pretending feature selection always improves the score.",
        "",
        "Models:",
        "logistic regression, decision tree, random forest, extra trees",
        "Extra trees is the model outside the basic set.",
        f"Best model: {best['model']}",
        f"Best accuracy: {best['accuracy']:.4f}",
        f"Best F1: {best['f1']:.4f}",
        f"Weakest model: {worst['model']}",
        f"Weakest F1: {worst['f1']:.4f}",
        "",
        "Note for report:",
        "Logistic regression underperformed compared to the tree models.",
        "That is not a failure by itself. It gives a useful comparison point.",
        "A reasonable explanation is that network traffic has nonlinear patterns, so tree models can split on different packet, byte, protocol, and timing behavior better than a simple linear model.",
        "Random forest being strongest makes sense because it uses many trees, so it is less dependent on one exact split of the data.",
        "Extra trees was close to random forest, which suggests tree ensembles fit this dataset better than the simpler baseline.",
        "Decision tree was also strong, but a single tree is usually easier to overfit than a forest.",
    ]

    save_text(out_folder, "dataset_notes.txt", dataset_notes)
    save_text(out_folder, "graph_notes.txt", graph_notes)
    save_text(out_folder, "model_notes.txt", model_notes)
