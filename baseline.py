import json
from collections import Counter

import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

from data import SharedTaskData, SharedTaskConstants


def evaluate_from_file(file_path: str):
    # Prepare storage of labels and dumpables
    y_true = {'novelty': [], 'validity': []}
    y_pred = {'novelty': [], 'validity': []}

    with open(file_path, "r") as f:
        predictions = json.load(f)

    for el in predictions:
        # Get true labels
        y_true['validity'].append(SharedTaskConstants.local_str_mapping[el['y_val']])
        y_true['novelty'].append(SharedTaskConstants.local_str_mapping[el['y_nov']])

        # Get true labels
        y_pred['validity'].append(SharedTaskConstants.local_str_mapping[el['pred_val']])
        y_pred['novelty'].append(SharedTaskConstants.local_str_mapping[el['pred_nov']])

    print_results(f"Evaluation results for file: {file_path}", y_true, y_pred)


def print_results(baseline_name: str, y_true: dict, y_pred: dict):
    print(f"==== {baseline_name} ====")
    print("Validity")
    results_validity = classification_report(
        y_true['validity'],
        y_pred['validity'],
        target_names=['not-valid', 'valid'],
        labels=[0, 1],
        zero_division=0
    )
    print(results_validity)

    print("Novelty")
    results_novelty = classification_report(
        y_true['novelty'],
        y_pred['novelty'],
        target_names=['not-novel', 'novel'],
        labels=[0, 1],
        zero_division=0
    )
    print(results_novelty)

    print("Combined (organization eval)")
    res = SharedTaskConstants.val_nov_metric(
        np.array(y_pred['validity']),
        np.array(y_true['validity']),
        np.array(y_pred['novelty']),
        np.array(y_true['novelty']),
    )
    print(res['f1_macro'].round(4))


def main():
    train_data = SharedTaskData("TaskA_train.csv")
    dev_data = SharedTaskData("TaskA_dev.csv")

    y_true = {'novelty': [], 'validity': []}
    y_pred_random = {'novelty': [], 'validity': []}
    y_pred_majority = {'novelty': [], 'validity': []}

    majority_val_label = Counter([y for _, y, _ in train_data]).most_common(1)[0][0]
    majority_nov_label = Counter([y for _, _, y in train_data]).most_common(1)[0][0]

    for sample, y_val, y_nov in tqdm(dev_data):
        y_true['validity'].append(SharedTaskConstants.local_str_mapping[y_val])
        y_true['novelty'].append(SharedTaskConstants.local_str_mapping[y_nov])
        y_pred_random['novelty'].append(np.random.randint(0, 2))
        y_pred_random['validity'].append(np.random.randint(0, 2))
        y_pred_majority['novelty'].append(SharedTaskConstants.local_str_mapping[majority_nov_label])
        y_pred_majority['validity'].append(SharedTaskConstants.local_str_mapping[majority_val_label])

    print_results('random', y_true, y_pred_random)
    print_results('majority', y_true, y_pred_majority)


if __name__ == "__main__":
    main()
