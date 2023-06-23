import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
import os
import pandas as pd

results_path = "C:/Users/vital/PycharmProjects/M12Project/prompt_engineering/results"


def check_assumptions(data1, data2, detector_name):
    # Check normality
    print(data1)
    print(data2)
    _, p_normality_data1 = shapiro(data1)
    _, p_normality_data2 = shapiro(data2)
    print(f"{detector_name} Normality Check")
    print(f"Normal: p = {p_normality_data1}")
    print(f"Modified: p = {p_normality_data2}")

    # Check homogeneity of variances
    _, p_homogeneity = levene(data1, data2)
    print(f"{detector_name} Homogeneity of Variances Check")
    print(f"p = {p_homogeneity}")

    # Check similarity of distributions
    plt.figure(figsize=(4, 3))
    sns.kdeplot(data1, label="Normal")
    sns.kdeplot(data2, label="Modified")
    plt.title(f"Distributions of {detector_name} Scores")
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f"{results_path}/{detector_name}_distributions.png")
    plt.close()

    plt.figure(figsize=(6, 3))
    sns.boxplot(data=[data1, data2], notch=False, medianprops={'color':'red'})
    plt.xticks([0, 1], ['Normal', 'Modified'])
    plt.title(f"Boxplot of {detector_name} Scores")
    plt.ylabel('Score (0-100 scale)')
    plt.savefig(f"{results_path}/{detector_name}_boxplot.png")
    plt.close()

    return p_normality_data1 > 0.05, p_normality_data2 > 0.05, p_homogeneity > 0.05


def perform_statistical_tests(data1, data2, detector_name):
    normal1, normal2, homogeneity = check_assumptions(data1, data2, detector_name)

    # Perform the appropriate test
    if normal1 and normal2 and homogeneity:
        # If data is normal and has equal variances, conduct t-test
        stat, p = ttest_ind(data1, data2)
        print(f'{detector_name} t-test results: Statistics={stat:.3f}, p={p:.3f}')
    else:
        # If data is not normal or doesn't have equal variances, conduct Mann-Whitney U test
        stat, p = mannwhitneyu(data1, data2)
        print(f'{detector_name} Mann-Whitney U test results: Statistics={stat:.3f}, p={p:.3f}')


def convert_labels_to_scores(labels):
    score_mapping = {
        "very unlikely": 5,
        "unlikely": 27.5,
        "unclear if it is": 67.5,
        "possibly": 94,
        "likely": 99
    }
    return [score_mapping[label] for label in labels]


def load_scores_from_csv(res_dir):
    csv_dir = res_dir
    detectors = ['openai', 'gpt2', 'turnitin']
    score_types = ['normal', 'smart']
    scores = {}
    for detector in detectors:
        for score_type in score_types:
            csv_file = f"gpt_4_scores_{detector}_{score_type}.csv"
            csv_path = os.path.join(csv_dir, csv_file)
            if os.path.exists(csv_path):
                with open(csv_path, 'r') as f:
                    line = f.readline().strip()
                    row_values = line.split(',')
                    if detector == 'openai':
                        scores[f"scores_{detector}_{score_type}"] = convert_labels_to_scores(row_values)
                    else:
                        # scale scores from 0-1 to 0-100
                        scores[f"scores_{detector}_{score_type}"] = [float(i)*100 if detector != 'turnitin' else float(i) for i in row_values]
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
    bar_width = 0.35
    index = np.arange(len(detector_names))

    for i, score_type in enumerate(score_types):
        plt.bar(index + i * bar_width, [avg_scores[f"{detector}_{score_type}"] for detector in detector_names],
                bar_width, label=score_type.capitalize())

    plt.xlabel('Detector')
    plt.ylabel('Scores (0-100 scale)')
    plt.title('"Smart prompt" vs "Baseline" for GPT4')
    plt.xticks(index + bar_width / 2, detector_names)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{results_path}/scores_histogram.png")
    plt.close()


def plot_boxplots(scores, detector_names, score_types):
    for detector in detector_names:
        plt.figure(figsize=(8, 6))
        data = [scores[f"scores_{detector}_{score_type}"] for score_type in score_types]
        plt.boxplot(data, notch=False, labels=score_types)

        plt.ylabel('Scores (0-100 scale)')
        plt.title(f'{detector.capitalize()} "Smart prompt" vs "Baseline" for GPT4')

        plt.tight_layout()
        plt.savefig(f"{results_path}/{detector}_scores_boxplot.png")
        plt.close()


def calculate_means(list1, list2):
    mean1 = round(sum(list1) / len(list1))
    mean2 = round(sum(list2) / len(list2))
    return mean1, mean2


scores = load_scores_from_csv(res_dir=results_path)
print(scores)
detector_names = ['openai', 'gpt2', 'turnitin']
score_types = ['normal', 'smart']

for detector in detector_names:
    perform_statistical_tests(scores[f"scores_{detector}_normal"], scores[f"scores_{detector}_smart"], detector)

plot_histograms(scores, detector_names, score_types)
plot_boxplots(scores, detector_names, score_types)

mean_scores_df = pd.DataFrame(index=detector_names, columns=score_types)

for detector in detector_names:
    mean_scores = calculate_means(scores[f"scores_{detector}_normal"], scores[f"scores_{detector}_smart"])
    mean_scores_df.loc[detector] = mean_scores
mean_scores_df = mean_scores_df.T
# Print DataFrame in LaTeX format
print(mean_scores_df.to_latex())