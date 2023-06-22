import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
import os
import csv

results_path = "C:/Users/vital/PycharmProjects/M12Project/advanced_prompt_engineering/perplexity_burstiness/results"


def plot_distribution_and_boxplot(data, detector_name):
    plt.figure(figsize=(20, 10))
    for category, scores in data.items():
        sns.kdeplot(scores, label=category)
    plt.title(f"Distributions of {detector_name} Scores")
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f"{results_path}/{detector_name}_distributions.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.boxplot(data=list(data.values()), notch=False,
                medianprops={'color': 'red'})
    plt.xticks(list(range(len(data.keys()))), list(data.keys()))  # You can also use an integer to set the size.
    plt.title(f"Boxplot of {detector_name} Scores")
    plt.ylabel('Score')
    plt.savefig(f"{results_path}/{detector_name}_boxplot.png")
    plt.close()


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
                        scores[score_type] = [score_mapping[i] if i in score_mapping else float(i) for i in row[1:]]
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
            print(score_type)
            print(scores)
            if detector == 'openai' or detector == 'gpt2':
                avg_scores[f"{detector}_{score_type}"] = np.mean(scores[f"{detector}_{score_type}"]) * 100
            else:
                avg_scores[f"{detector}_{score_type}"] = np.mean(scores[f"{detector}_{score_type}"])

    labels = list(avg_scores.keys())
    values = list(avg_scores.values())

    # plot setup
    fig, ax = plt.subplots()
    bar_width = 0.22
    index = np.arange(len(detector_names) * len(score_types))

    for i, detector in enumerate(detector_names):
        plt.bar(index[i * len(score_types):(i + 1) * len(score_types)],
                [avg_scores[f"{detector}_{score_type}"] for score_type in score_types], bar_width, label=detector)

    plt.xlabel('Detector')
    plt.ylabel('Scores (0-100 scale)')
    plt.title('Comparison of different prompt types')
    plt.xticks(index, labels, rotation='vertical')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{results_path}/scores_histogram.png")
    plt.close()


scores = load_scores_from_csv()
detectors = ['openai', 'gpt2', 'turnitin']
# score_types = ['standard', 'explanation', 'perplexity', 'burstiness', 'both', 'regen_standard', 'regen_explanation', 'regen_perplexity', 'regen_burst', 'regen_both']

# Renaming rules
score_type_mapping = {
    'standard': 'std',
    'explanation': 'exp',
    'perplexity': 'plx',
    'burstiness': 'bst',
    'both': 'bth',
    'regen_standard': 'rgn_std',
    'regen_explanation': 'rgn_exp',
    'regen_perplexity': 'rgn_plx',
    'regen_burst': 'rgn_bst',
    'regen_both': 'rgn_bth'
}

# Get a list of all the types of scores for later usage
score_types = list(score_type_mapping.values())

# Create a new dictionary with the new names
adjusted_scores = {}

# Iterate over the original dictionary
for k, v in scores.items():
    # Remove prefix and split into detector and score type
    prefix, detector, *score_type = k.split('_')

    # Join score_type elements back into a single string
    score_type = "_".join(score_type)

    # Rename score_type using mapping dictionary
    new_score_type = score_type_mapping.get(score_type, score_type)

    # Create the new key and update the dictionary
    new_key = '_'.join([detector, new_score_type])
    adjusted_scores[new_key] = v

print(adjusted_scores)

for detector in detectors:
    # create a subset of the scores dictionary for each detector
    detector_scores = {"_".join(k.split("_")[1:]): v for k, v in adjusted_scores.items() if k.startswith(detector)}
    print(detector_scores)
    # plot the distribution and boxplot for each detector
    plot_distribution_and_boxplot(detector_scores, detector)

# plot the histograms for all detectors and score types
plot_histograms(adjusted_scores, detectors, score_types)
