import scipy.stats as stats
from anova_detectors import load_detector_results_from_csv, load_scores_turnitin

results_directory = "C:/Users/vital/PycharmProjects/M12Project/style_diff_test/results"

scores_gpt2 = load_detector_results_from_csv('GPT2 output classifier', results_directory)
scores_turnitin = load_scores_turnitin('Turnitin', results_directory)


def check_assumptions(scores_dict):
    # Check Normality
    normality_met = True
    homogeneity_met = True
    for style, scores in scores_dict.items():
        _, p = stats.shapiro(scores)
        if p > 0.05:
            print(f"{style} follows Normal Distribution")
        else:
            print(f"{style} does not follow Normal Distribution")
            normality_met = False

    # Check Homogeneity of variances
    _, p = stats.levene(*scores_dict.values())
    if p > 0.05:
        print("Variances are equal across all groups")
    else:
        print("Variances are not equal across all groups")
        homogeneity_met = False

    # Return whether assumptions are met
    return normality_met and homogeneity_met


def anova_test(scores_dict):
    # Check assumptions
    if check_assumptions(scores_dict):
        # Perform ANOVA
        _, p = stats.f_oneway(*scores_dict.values())
        print("ANOVA p-value:", p)
    else:
        print("Assumptions not met. Cannot perform ANOVA.")

print(scores_turnitin)
print("GPT2 scores")
anova_test(scores_gpt2)
print("\nTurnitin scores")
anova_test(scores_turnitin)
