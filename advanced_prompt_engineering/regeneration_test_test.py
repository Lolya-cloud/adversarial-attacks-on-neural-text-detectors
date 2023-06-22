import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
import os
import csv

results_path = "C:/Users/vital/PycharmProjects/M12Project/advanced_prompt_engineering/results"

def convert_labels_to_scores(labels):
    score_mapping = {
        "very unlikely": 0.05,
        "unlikely": 0.275,
        "unclear if it is": 0.675,
        "possibly": 0.94,
        "likely": 0.99
    }
    return [score_mapping[label] for label in labels]


def check_assumptions(data, detector_name):
    print(f"{detector_name} Normality Check")
    for category, scores in data.items():
        _, p_normality = shapiro(scores)
        print(f"{category}: p = {p_normality}")

    plt.figure(figsize=(12, 6))
    for category, scores in data.items():
        sns.kdeplot(scores, label=category)
    plt.title(f"Distributions of {detector_name} Scores")
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f"{results_path}/{detector_name}_distributions.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=list(data.values()), notch=False, medianprops={'color':'red'})
    plt.xticks([0, 1, 2, 3], list(data.keys()))
    plt.title(f"Boxplot of {detector_name} Scores")
    plt.ylabel('Score')
    plt.savefig(f"{results_path}/{detector_name}_boxplot.png")
    plt.close()


"""def load_scores_from_csv():
    detectors = ['openai', 'gpt2', 'turnitin']
    score_types = ['normal', 'smart', 'normal_regen', 'smart_regen']
    scores = {}
    for detector in detectors:
        for score_type in score_types:
            csv_file = f"scores_{detector}_{score_type}.csv"
            csv_path = os.path.join(results_path, csv_file)
            if os.path.exists(csv_path):
                with open(csv_path, 'r') as f:
                    line = f.readline().strip()
                    row_values = line.split(',')
                    if detector == 'openai':
                        scores[f"scores_{detector}_{score_type}"] = convert_labels_to_scores(row_values)
                    else:
                        # scale scores from 0-1 to 0-100
                        scores[f"scores_{detector}_{score_type}"] = [float(i)*100 for i in row_values]
            else:
                print(f"File {csv_path} does not exist.")
    return scores"""


def load_scores_from_csv():
    csv_dir = results_path
    detectors = ['openai', 'gpt2', 'turnitin']
    scores = {}

    score_mapping = {
        "very unlikely": 0.05,
        "unlikely": 0.275,
        "unclear if it is": 0.675,
        "possibly": 0.94,
        "likely": 0.99
    }

    for detector in detectors:
        csv_file = f"{detector}_scores.csv"
        csv_path = os.path.join(csv_dir, csv_file)
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    score_type = row[0]
                    if detector == 'openai':
                        scores[score_type] = [score_mapping[i] for i in row[1:]]
                    else:
                        scores[score_type] = [float(i) for i in row[1:]]
        else:
            print(f"File {csv_path} does not exist.")
    return scores


def plot_histograms(scores, detector_names, score_types):
    avg_scores = {}
    # calculate average scores
    for detector in detector_names:
        for score_type in score_types:
            avg_scores[f"{detector}_{score_type}"] = np.mean(scores[f"scores_{detector}_{score_type}"])

    labels = list(avg_scores.keys())
    values = list(avg_scores.values())

    # plot setup
    fig, ax = plt.subplots()
    bar_width = 0.22
    index = np.arange(len(detector_names))

    for i, score_type in enumerate(score_types):
        plt.bar(index + i * bar_width, [avg_scores[f"{detector}_{score_type}"] for detector in detector_names],
                bar_width, label=score_type.capitalize())

    plt.xlabel('Detector')
    plt.ylabel('Scores (0-100 scale)')
    plt.title('Comparison of different prompt types')
    plt.xticks(index + bar_width * 1.5, detector_names)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{results_path}/scores_histogram.png")
    plt.close()


def plot_boxplots(scores, detector_names, score_types):
    for detector in detector_names:
        plt.figure(figsize=(8, 6))
        data = [scores[f"scores_{detector}_{score_type}"] for score_type in score_types]
        plt.boxplot(data, notch=False, medianprops={'color':'red'}, labels=[name.capitalize() for name in score_types])

        plt.ylabel('Scores (0-100 scale)')
        plt.title(f'{detector.capitalize()} score comparison')

        plt.tight_layout()
        plt.savefig(f"{results_path}/{detector}_scores_boxplot.png")
        plt.close()


scores = load_scores_from_csv()
print(scores)
detector_names = ['openai', 'gpt2', 'turnitin']
score_types = ['normal', 'smart', 'normal_regen', 'smart_regen']

for detector in detector_names:
    data = {f"{score_type}": scores[f"scores_{detector}_{score_type}"] for score_type in score_types}
    check_assumptions(data, detector)

plot_histograms(scores, detector_names, score_types)
plot_boxplots(scores, detector_names, score_types)
