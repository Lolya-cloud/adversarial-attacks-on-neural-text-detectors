from anova_detectors import load_detector_results_from_csv, load_scores_turnitin
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

results_directory = "C:/Users/vital/PycharmProjects/M12Project/style_diff_test/results"

scores_gpt2 = load_detector_results_from_csv('GPT2 output classifier', results_directory)
scores_turnitin = load_scores_turnitin('Turnitin', results_directory)
scores_openai = load_detector_results_from_csv('OpenaAI classifier', results_directory)


def plot_data_num(data_dict, res_dir, detector_name):
    # Flatten the dictionary into a list of scores
    scores = [score for sublist in data_dict.values() for score in sublist]

    # Transform the list into a DataFrame
    df = pd.DataFrame(scores, columns=['Score'])

    # Create the violin plot
    plt.figure(figsize=(12, 8))
    sns.violinplot(y='Score', data=df)
    plt.title('Distribution of Scores Across All Essay Styles')
    plt.tight_layout()

    # Save the plot to the specified directory
    plt.savefig(f"{res_dir}/{detector_name}_performance.png")
    plt.close()


def plot_data_label(data_dict, res_dir, detector_name):
    score_mapping = {
        "very unlikely": 0.05,
        "unlikely": 0.275,
        "unclear if it is": 0.675,
        "possibly": 0.94,
        "likely": 0.99
    }

    # Flatten the dictionary and map labels to scores
    scores = [score_mapping[label] for sublist in data_dict.values() for label in sublist]

    # Transform the list into a DataFrame
    df = pd.DataFrame(scores, columns=['Score'])

    # Create the violin plot
    plt.figure(figsize=(12, 8))
    sns.violinplot(y='Score', data=df)
    plt.title('Distribution of Scores Across All Essay Styles')
    plt.tight_layout()

    # Save the plot to the specified directory
    plt.savefig(f"{res_dir}/{detector_name}_performance.png")
    plt.close()


plot_data_num(scores_gpt2, results_directory, "GPT2 output classifier")
plot_data_num(scores_turnitin, results_directory, "Turnitin")
plot_data_label(scores_openai, results_directory, "OpenAI classifier")

