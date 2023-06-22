from generators.gpt3_5_turbo import ChatGPT
from generators.openAiParameters import Parameters
from detectors.open_ai_classifier import OpenAiClassifier
from detectors.turnitin_detector import TurnitIn
from detectors.gpt_2_detector import GPT2Detector
from style_diff_test.anova_detectors import AnovaDetectors
import os
import matplotlib.pyplot as plt
import csv
import pickle
import threading

textPromptRef = "Write a five-hundred-words(500) essay on the topic 'Should school students be allowed " \
                "to curate their high school curriculum?'." \
                "Include personal reflections, use a mix of long and short sentences, " \
                "employ rhetorical questions to engage the reader, maintain a conversational tone in parts, " \
                "and play around with the paragraph structure to create a dynamic and engaging piece of writing."
textPromptNorm = "Write a five-hundred-words(500) essay on the topic 'Should school students be allowed to " \
                 "curate their high school curriculum?'."

iterations = 20
generator = ChatGPT()

texts_dir = "C:/Users/vital/PycharmProjects/M12Project/texts"
prompts_list_normal = []
prompts_list_smart = []
filepaths_list_normal = []
filepaths_list_smart = []
parameters = Parameters(max_tokens=800)

pickle_file = 'data.pickle'
generation_data = {}
if os.path.exists(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        generation_data = data['generation']
        prompts_list_normal = generation_data['prompts_list_normal']
        filepaths_list_normal = generation_data['filepaths_list_normal']
        prompts_list_smart = generation_data['prompts_list_smart']
        filepaths_list_smart = generation_data['filepaths_list_smart']
else:
    def generate_normal():
        for i in range(0, iterations):
            file_name = f"standard_prompt_iteration_{i}.txt"
            print(f"Generating {file_name}; prompt: {textPromptNorm}")
            prompts_list_normal.append(textPromptNorm)
            filepaths_list_normal.append(os.path.join(texts_dir, file_name))
            generator.generate_prompt_save_txt_bounds(textPromptNorm, parameters=parameters, dir_name=texts_dir,
                                                      file_name=file_name, lower_bound=425, upper_bound=575)


    def generate_smart():
        for j in range(0, iterations):
            parameters = Parameters(max_tokens=800)
            file_name = f"new_prompt_iteration_{j}.txt"
            print(f"Generating {file_name}; prompt: {textPromptRef}")
            prompts_list_normal.append(textPromptRef)
            filepaths_list_smart.append(os.path.join(texts_dir, file_name))
            generator.generate_prompt_save_txt_bounds(textPromptRef, parameters, texts_dir, file_name,
                                                      lower_bound=425, upper_bound=575)


    thread1 = threading.Thread(target=generate_normal)
    thread2 = threading.Thread(target=generate_smart)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    generation_data = {
        'prompts_list_normal': prompts_list_normal,
        'filepaths_list_normal': filepaths_list_normal,
        'prompts_list_smart': prompts_list_smart,
        'filepaths_list_smart': filepaths_list_smart,
    }

    with open(pickle_file, 'wb') as f:
        pickle.dump({'generation': generation_data}, f)

detectors = AnovaDetectors(turnitin=0, openai_classifier=0, generator=ChatGPT(),
                           gpt2_detector=GPT2Detector())

print("Turnitin analysis")
detectors.turnitin = TurnitIn()
scores_turntin_normal, null_scores_normal = detectors.calculate_scores_turnitin(filepaths_list_normal, prompts_list_normal, parameters, 30, 15)
scores_turntin_smart, null_scores_smart = detectors.calculate_scores_turnitin(filepaths_list_smart, prompts_list_smart, parameters, 30, 15)
detectors.turnitin.close()
print(filepaths_list_normal, prompts_list_normal)
print("Openai analysis")
detectors.openai_classifier = OpenAiClassifier()
scores_openai_normal = detectors.calculate_scores_openai(filepaths_list_normal)
scores_openai_smart = detectors.calculate_scores_openai(filepaths_list_smart)
detectors.openai_classifier.close()
print("GPT2 analysis")
scores_gpt2_normal = detectors.calculate_scores_gpt2(filepaths_list_normal)
scores_gpt2_smart = detectors.calculate_scores_gpt2(filepaths_list_smart)


data = {
    'scores_openai_normal': scores_openai_normal,
    'scores_openai_smart': scores_openai_smart,
    'scores_gpt2_normal': scores_gpt2_normal,
    'scores_gpt2_smart': scores_gpt2_smart,
    'scores_turnitin_normal': scores_turntin_normal,
    'scores_turnitin_smart': scores_turntin_smart,
}

with open(pickle_file, 'wb') as f:
    pickle.dump({'generation': generation_data, 'scores': data}, f)

csv_dir = "C:/Users/vital/PycharmProjects/M12Project/basic_prompt_engineering_results"
os.makedirs(csv_dir, exist_ok=True)
for key, value in data.items():
    with open(os.path.join(csv_dir, f"{key}.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(value)

detectors = ['openai', 'gpt2', 'turnitin']

for detector in detectors:
    plt.figure()
    # Explicitly specifying the x-values as a range of integers
    plt.plot(range(iterations), data[f'scores_{detector}_normal'], label='Normal')
    plt.plot(range(iterations), data[f'scores_{detector}_smart'], label='Smart')
    plt.title(f'Comparison of Normal and Smart scores for {detector}')
    plt.xlabel('Iteration')
    plt.ylabel('Score')

    # If the detector is 'openai', reverse the y-axis
    if detector == 'openai':
        plt.gca().invert_yaxis()

    plt.legend()
    plt.savefig(os.path.join(csv_dir, f"{detector}_comparison_plot.png"))

plt.show()
