import time

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


texts_dir = "C:\\Users\\vital\\PycharmProjects\\M12Project\\prompt_engineering\\texts"
detectors = AnovaDetectors(turnitin=0, openai_classifier=0, generator=ChatGPT(),
                           gpt2_detector=GPT2Detector())

parameters = Parameters(max_tokens=800)

textPromptRef = "Write a five-hundred-words(500) essay on the topic 'Should school students be allowed to curate their high school curriculum?'. Include personal reflections, use a mix of long and short sentences, employ rhetorical questions to engage the reader, maintain a conversational tone in parts, and play around with the paragraph structure to create a dynamic and engaging piece of writing."
textPromptNorm = "Write a five-hundred-words(500) essay on the topic 'Should school students be allowed to curate their high school curriculum?'."

filepaths_list_normal = []
filepaths_list_smart = []
prompts_list_normal = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
prompts_list_smart = prompts_list_normal

for i in range(0, 10):
    filename = os.path.join(texts_dir, f"normal_{i}.txt")
    filepaths_list_normal.append(filename)
    filename_smart = os.path.join(texts_dir, f"smart_{i}.txt")
    filepaths_list_smart.append(filename_smart)

print(filepaths_list_normal)
print(filepaths_list_smart)

print("Turnitin analysis")
detectors.turnitin = TurnitIn()
scores_turntin_normal, null_scores_normal = detectors.calculate_scores_turnitin(filepaths_list_normal, prompts_list_normal, parameters, 30, 15)
scores_turntin_smart, null_scores_smart = detectors.calculate_scores_turnitin(filepaths_list_smart, prompts_list_smart, parameters, 30, 15)
detectors.turnitin.close()
print("Openai analysis")
detectors.openai_classifier = OpenAiClassifier()
scores_openai_normal = detectors.calculate_scores_openai(filepaths_list_normal)
scores_openai_smart = detectors.calculate_scores_openai(filepaths_list_smart)
detectors.openai_classifier.close()
print("GPT2 analysis")
scores_gpt2_normal = detectors.calculate_scores_gpt2(filepaths_list_normal)
scores_gpt2_smart = detectors.calculate_scores_gpt2(filepaths_list_smart)


data = {
    'gpt_4_scores_openai_normal': scores_openai_normal,
    'gpt_4_scores_openai_smart': scores_openai_smart,
    'gpt_4_scores_gpt2_normal': scores_gpt2_normal,
    'gpt_4_scores_gpt2_smart': scores_gpt2_smart,
    'gpt_4_scores_turnitin_normal': scores_turntin_normal,
    'gpt_4_scores_turnitin_smart': scores_turntin_smart,
}

csv_dir = "C:/Users/vital/PycharmProjects/M12Project/prompt_engineering/results"
os.makedirs(csv_dir, exist_ok=True)
for key, value in data.items():
    with open(os.path.join(csv_dir, f"{key}.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(value)

detectors = ['openai', 'gpt2', 'turnitin']
