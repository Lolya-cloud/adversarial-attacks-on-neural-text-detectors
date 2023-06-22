import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from optimizer import Optimizer
from generators.gpt3_5_turbo import ChatGPT
import matplotlib.colorbar
import os


prompt_train = "Write a formal five-hundred-word argumentative essay on the topic 'Should school students be allowed " \
               "to curate their high school curriculum?'."
dir_res = "C:\\Users\\vital\\PycharmProjects\\M12Project\\parameter_optimization\\optimization\\results"
dir_texts = "C:\\Users\\vital\\PycharmProjects\\M12Project\\parameter_optimization\\optimization\\texts"
generator = ChatGPT()
optimizer = Optimizer(dir_texts, dir_res, generator)

csv_file = os.path.join(dir_res, 'dataframe.csv')
data = optimizer.load_dataframe()

if data is None:
    print("Generating data")
    data = optimizer.generate_many_save_txt(prompt_train, 1)
    csv_file = optimizer.save_dataframe(data)

if 'gpt2_scores' not in data.columns:
    print("gpt 2 analysis")
    csv_file = optimizer.analyse_gpt2(data)

if 'openai_scores' not in data.columns:
    print("openai analysis")
    csv_file = optimizer.analyse_openai(data)

if 'turnitin_scores' not in data.columns:
    print("turnitin analysis")
    csv_file = optimizer.analyse_turnitin(data, prompt_train)

data = pd.read_csv(csv_file)


def plot_results(df, detector_name, save_as, label):
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['frequency_penalty'], df['presence_penalty'], c=df[detector_name], cmap='viridis')
    plt.title(label)
    plt.xlabel('Frequency Penalty')
    plt.ylabel('Presence Penalty')
    cbar = plt.colorbar(scatter)
    cbar.set_label(label + ' Scores')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_res, save_as))

# Plot for each detector
detectors = [('gpt2_scores', 'GPT2 Output Detector'),
             ('openai_scores', 'OpenAI Classifier'),
             ('turnitin_scores', 'Turnitin')]

for detector, title in detectors:
    plot_results(data, detector, f"{detector}_plot.png", title)

# Combined plot of three previous
fig, axs = plt.subplots(1, 3, figsize=(10, 3))
for ax, (detector, title) in zip(axs, detectors):
    scatter = ax.scatter(data['frequency_penalty'], data['presence_penalty'], c=data[detector], cmap='viridis')
    ax.set_title(title + " Scores")
    ax.set_xlabel('Frequency Penalty')
    ax.set_ylabel('Presence Penalty')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(title + ' Scores')
plt.tight_layout()
plt.savefig(os.path.join(dir_res, 'combined_side_by_side_plot.png'))

# Plot with sum of scores (converted to integers)
data['total_scores'] = data['gpt2_scores']*100 + data['openai_scores']*100 + data['turnitin_scores']
plot_results(data, 'total_scores', 'total_scores_plot.png', 'Total')

# Selecting the rows with the minimum score for each detector
# best_gpt2 = data[data['gpt2_scores'] == data['gpt2_scores'].min()]
best_openai = data[data['openai_scores'] == data['openai_scores'].min()]
best_turnitin = data[data['turnitin_scores'] == data['turnitin_scores'].min()]

# Sorting the DataFrame by GPT2 scores and selecting the top 10
best_gpt2 = data.sort_values('gpt2_scores', ascending=True).head(10)

print("10 best parameter sets for GPT2 Output Detector:")
print(best_gpt2[['frequency_penalty', 'presence_penalty', 'gpt2_scores']])


print("Best parameter sets for OpenAI Classifier:")
print(best_openai[['frequency_penalty', 'presence_penalty', 'openai_scores']])

print("Best parameter sets for Turnitin:")
print(best_turnitin[['frequency_penalty', 'presence_penalty', 'turnitin_scores']])
