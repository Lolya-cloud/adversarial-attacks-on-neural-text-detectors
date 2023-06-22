import matplotlib.pyplot as plt
import scipy.stats as stats
from anova_detectors import load_detector_results_from_csv, load_scores_turnitin
from scipy.stats import skew

results_directory = "C:/Users/vital/PycharmProjects/M12Project/style_diff_test/results"

scores_gpt2 = load_detector_results_from_csv('GPT2 output classifier', results_directory)
scores_turnitin = load_scores_turnitin('Turnitin', results_directory)
scores_openai = load_detector_results_from_csv("OpenAI classifier", results_directory)

print(scores_gpt2)


def check_assumptions(scores_dict):
    # Create boxplots for each group
    for style, scores in scores_dict.items():
        plt.boxplot(scores, positions=[1], widths=0.6)
        plt.title(style)
        plt.show()
    # Check skewness for each group
    for style, scores in scores_dict.items():
        print(f'Skewness for {style} is {skew(scores)}')


def perform_test(scores_dict):
    # Perform Kruskal-Wallis H test
    _, p = stats.kruskal(*scores_dict.values())
    print("Kruskal-Wallis H test p-value:", p)


check_assumptions(scores_gpt2)
perform_test(scores_gpt2)