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

textPromptNorm = "Write a five-hundred-words(500) essay on the topic 'Should school students be allowed to curate their high school curriculum?'."
textPromptSmart = "Write a five-hundred-words(500) essay on the topic 'Should school students be allowed to curate their high school curriculum?'. Include personal reflections, use a mix of long and short sentences, employ rhetorical questions to engage the reader, maintain a conversational tone in parts, and play around with the paragraph structure to create a dynamic and engaging piece of writing. Try to include factual and contextual information, use advanced concepts and vocabulary. Utilize a combination of complex and simple vocabulary. Try to mimic human writing as closely as you can. Avoid passive voice, as it tends to occur more often in AI-generated texts. Add a few examples from the real world illustrating your point."
textPromptNormReg = "Regenerate the essay. Make sure to fit in five-hundred words."
textPromptSmartReg = "Regenerate the essay. Write a five-hundred-words(500) essay on the topic 'Should school students be allowed to curate their high school curriculum?'. Include personal reflections, use a mix of long and short sentences, employ rhetorical questions to engage the reader, maintain a conversational tone in parts, and play around with the paragraph structure to create a dynamic and engaging piece of writing. Try to include factual and contextual information, use advanced concepts and vocabulary. Utilize a combination of complex and simple vocabulary. Try to mimic human writing as closely as you can. Avoid passive voice, as it tends to occur more often in AI-generated texts. Add a few examples from the real world illustrating your point."

iterations = 10
generator = ChatGPT()

texts_dir_norm_once = "C:/Users/vital/PycharmProjects/M12Project/advanced_prompt_engineering/texts/baseline"
texts_dir_smart_once = "C:/Users/vital/PycharmProjects/M12Project/advanced_prompt_engineering/texts/baseline_improved"
texts_dir_norm_regen = "C:/Users/vital/PycharmProjects/M12Project/advanced_prompt_engineering/texts/regen_baseline"
texts_dir_smart_regen = "C:/Users/vital/PycharmProjects/M12Project/advanced_prompt_engineering/texts/regen_improved"
res_dir = "C:/Users/vital/PycharmProjects/M12Project/advanced_prompt_engineering/results"

prompts_list_normal = []
prompts_list_smart = []
prompts_list_normal_regen = []
prompts_list_smart_regen = []

filepaths_list_normal = []
filepaths_list_smart = []
filepaths_list_normal_regen = []
filepaths_list_smart_regen = []

parameters = Parameters(max_tokens=800)

for i in range(0, iterations):
    prompts_list_normal.append(textPromptNorm)
    prompts_list_smart.append(textPromptSmart)
    prompts_list_normal_regen.append(textPromptNormReg)
    prompts_list_smart_regen.append(textPromptSmartReg)

pickle_file = 'adv_data.pickle'
generation_data = {}
if os.path.exists(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        generation_data = data['generation']
        filepaths_list_normal = generation_data['filepaths_list_normal']
        filepaths_list_smart = generation_data['filepaths_list_smart']
        filepaths_list_normal_regen = generation_data['filepaths_list_normal_reg']
        filepaths_list_smart_regen = generation_data['filepaths_list_smart_reg']
else:
    def generate_normal():
        for j in range(0, iterations):
            file_name = f"standard_prompt_iteration_{j}.txt"
            print(f"Generating {file_name}; prompt: {textPromptNorm}")
            filepaths_list_normal.append(os.path.join(texts_dir_norm_once, file_name))
            generator.generate_prompt_save_txt_bounds(textPromptNorm, parameters=parameters, dir_name=texts_dir_norm_once,
                                                      file_name=file_name, lower_bound=375, upper_bound=625)

    def generate_smart():
        for j in range(0, iterations):
            file_name = f"smart_prompt_iteration_{j}.txt"
            print(f"Generating {file_name}; prompt: {textPromptSmart}")
            filepaths_list_smart.append(os.path.join(texts_dir_smart_once, file_name))
            generator.generate_prompt_save_txt_bounds(textPromptSmart, parameters, texts_dir_smart_once, file_name,
                                                      lower_bound=375, upper_bound=625)

    def generate_regen():
        for j in range(0, iterations):
            file_name = f"norm_regen_prompt_iteration_{j}.txt"
            print(f"Generating {file_name}; first prompt: {textPromptNorm}, second prompt: {textPromptNormReg}")
            filepaths_list_normal_regen.append(os.path.join(texts_dir_norm_regen, file_name))
            generator.generate_text_refine_prompts_save_txt_bounds(first_prompt=textPromptNorm,
                                                                   second_prompt=textPromptNormReg,
                                                                   dir_name=texts_dir_norm_regen,
                                                                   file_name=file_name,
                                                                   parameters=parameters,
                                                                   lower_bound=375,
                                                                   upper_bound=625)

    def generate_smart_regen():
        for j in range(0, iterations):
            file_name = f"smart_regen_prompt_iteration_{j}.txt"
            print(f"Generating {file_name}; first prompt: {textPromptNorm}, second prompt: {textPromptSmartReg}")
            filepaths_list_smart_regen.append(os.path.join(texts_dir_smart_regen, file_name))
            generator.generate_text_refine_prompts_save_txt_bounds(first_prompt=textPromptNorm,
                                                                   second_prompt=textPromptSmartReg,
                                                                   dir_name=texts_dir_smart_regen,
                                                                   file_name=file_name,
                                                                   parameters=parameters,
                                                                   lower_bound=375,
                                                                   upper_bound=625)

    thread1 = threading.Thread(target=generate_normal)
    thread2 = threading.Thread(target=generate_smart)
    thread3 = threading.Thread(target=generate_regen)
    thread4 = threading.Thread(target=generate_smart_regen)

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()

    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()

    generation_data = {
        'filepaths_list_normal': filepaths_list_normal,
        'filepaths_list_smart': filepaths_list_smart,
        'filepaths_list_normal_reg': filepaths_list_normal_regen,
        'filepaths_list_smart_reg': filepaths_list_smart_regen
    }

    with open(pickle_file, 'wb') as f:
        pickle.dump({'generation': generation_data}, f)

detectoring = AnovaDetectors(turnitin=0, openai_classifier=0, generator=ChatGPT(),
                           gpt2_detector=GPT2Detector())

detectors = ['openai', 'gpt2', 'turnitin']

for detector in detectors:
    detector_csv_path = os.path.join(res_dir, f"{detector}_scores.csv")

    # Skip the detector if the corresponding CSV file already exists.
    if os.path.exists(detector_csv_path):
        print(f"Skipping {detector} analysis as its CSV file already exists.")
        continue

    print(f"{detector.capitalize()} analysis")

    if detector == "openai":
        detectoring.openai_classifier = OpenAiClassifier()
        scores_openai_normal = detectoring.calculate_scores_openai(filepaths_list_normal)
        scores_openai_smart = detectoring.calculate_scores_openai(filepaths_list_smart)
        scores_openai_normal_regen = detectoring.calculate_scores_openai(filepaths_list_normal_regen)
        scores_openai_smart_regen = detectoring.calculate_scores_openai(filepaths_list_smart_regen)
        detectoring.openai_classifier.close()

        detector_data = {
            'scores_openai_normal': scores_openai_normal,
            'scores_openai_smart': scores_openai_smart,
            'scores_openai_normal_regen': scores_openai_normal_regen,
            'scores_openai_smart_regen': scores_openai_smart_regen,
        }

    elif detector == "gpt2":
        scores_gpt2_normal = detectoring.calculate_scores_gpt2(filepaths_list_normal)
        scores_gpt2_smart = detectoring.calculate_scores_gpt2(filepaths_list_smart)
        scores_gpt2_normal_regen = detectoring.calculate_scores_gpt2(filepaths_list_normal_regen)
        scores_gpt2_smart_regen = detectoring.calculate_scores_gpt2(filepaths_list_smart_regen)

        detector_data = {
            'scores_gpt2_normal': scores_gpt2_normal,
            'scores_gpt2_smart': scores_gpt2_smart,
            'scores_gpt2_normal_regen': scores_gpt2_normal_regen,
            'scores_gpt2_smart_regen': scores_gpt2_smart_regen,
        }

    elif detector == "turnitin":
        detectoring.turnitin = TurnitIn()
        scores_turntin_normal, null_scores_normal = detectoring.calculate_scores_turnitin(filepaths_list_normal,
                                                                                        prompts_list_normal, parameters,
                                                                                        60, 15)
        scores_turntin_smart, null_scores_smart = detectoring.calculate_scores_turnitin(filepaths_list_smart,
                                                                                      prompts_list_smart, parameters,
                                                                                      60, 15)
        scores_turntin_normal_regen, null_scores_normal_regen = detectoring.calculate_scores_turnitin(
            filepaths_list_normal_regen, prompts_list_normal_regen, parameters, 60, 15)
        scores_turntin_smart_regen, null_scores_smart_regen = detectoring.calculate_scores_turnitin(
            filepaths_list_smart_regen, prompts_list_smart_regen, parameters, 60, 15)
        detectoring.turnitin.close()

        detector_data = {
            'scores_turnitin_normal': scores_turntin_normal,
            'scores_turnitin_smart': scores_turntin_smart,
            'scores_turnitin_normal_regen': scores_turntin_normal_regen,
            'scores_turnitin_smart_regen': scores_turntin_smart_regen,
        }

    # Save detector's scores to CSV
    with open(detector_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in detector_data.items():
            writer.writerow([key] + value)
