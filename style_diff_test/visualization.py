from anova_detectors import load_detector_results_from_csv, load_scores_turnitin
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Define the directory where the results will be saved
results_directory = "C:/Users/vital/PycharmProjects/M12Project/style_diff_test/results"

# Load the scores from the CSV files
scores_gpt2 = load_detector_results_from_csv('GPT2 output classifier', results_directory)
scores_turnitin = load_scores_turnitin('Turnitin', results_directory)
scores_openai = load_detector_results_from_csv("OpenaAI classifier", results_directory)

# Convert OpenAI Classifier scores to a numeric scale
score_mapping = {"very unlikely": 0.05, "unlikely": 0.275, "unclear if it is": 0.675, "possibly": 0.94, "likely": 0.99}
scores_openai = {key: [score_mapping[i] for i in val] for key, val in scores_openai.items()}

# Create separate plots for each detector
detectors = [('Turnitin', scores_turnitin), ('OpenAI', scores_openai), ('GPT2', scores_gpt2)]
short_labels = {
    "argumentative": "arg",
    "cause_and_effect": "cause",
    "compare_contrast": "compare",
    "controversial_argumentative": "contr_arg",
    "descriptive": "desc",
    "expository": "expository",
    "funny_argumentative": "funny_arg",
    "narrative": "narrative",
    "persuasive": "persuasive",
    "research": "research"
    # Add more mappings as needed
}
for detector_name, detector_scores in detectors:
    # Calculate means for each label
    means = {key: np.mean(val) for key, val in detector_scores.items()}
    labels = list(means.keys())
    means = list(means.values())

    # Create a histogram of mean scores for each label
    plt.figure(figsize=(10, 8))  # Decrease the width of the plot
    plt.bar([short_labels.get(label, label) for label in labels], means)
    plt.title(f'Histogram of Mean Scores for {detector_name}')
    plt.xlabel('Label')
    plt.ylabel('Mean Score')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, f'{detector_name}_histogram.png'))
    plt.close()

    # Prepare data for boxplot
    scores_df = pd.DataFrame.from_dict(detector_scores, orient='index').transpose()

    # Replace column names with shorter labels for boxplot
    scores_df.columns = [short_labels.get(col, col) for col in scores_df.columns]

    # Create a boxplot of scores for each label
    plt.figure(figsize=(5, 4))  # Decrease the width of the plot
    sns.boxplot(data=scores_df, medianprops={'color':'red'})
    plt.title(f'Boxplot of Scores for {detector_name}')
    plt.xlabel('Label')
    plt.ylabel('Score')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, f'{detector_name}_boxplot.png'))
    plt.close()
