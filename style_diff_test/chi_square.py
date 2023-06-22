import scipy.stats as stats
from anova_detectors import load_detector_results_from_csv
import numpy as np
import pandas as pd

results_directory = "C:/Users/vital/PycharmProjects/M12Project/style_diff_test/results"
scores_openai = load_detector_results_from_csv('OpenaAI classifier', results_directory)
print(scores_openai)
# Create a contingency table
contingency_table = pd.crosstab(
    pd.Series([item for sublist in scores_openai.values() for item in sublist], name='labels'),
    pd.Series([key for key, sublist in scores_openai.items() for item in sublist], name='essay_style')
)

# Check assumptions: All expected frequencies should be 5 or greater
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

if np.all(expected >= 5):
    print("All expected frequencies are 5 or greater. Assumption met.")
    # Perform Chi-square test
    print("Chi-square p-value:", p)
else:
    print("Some expected frequencies are less than 5. Assumption not met.")
    # Identify cells where expected frequency is less than 5
    problematic_cells = np.argwhere(expected < 5)
    for cell in problematic_cells:
        row_index, col_index = cell
        essay_style = contingency_table.columns[col_index]
        label = contingency_table.index[row_index]
        print(f"Expected frequency for label '{label}' in essay style '{essay_style}' is less than 5.")


