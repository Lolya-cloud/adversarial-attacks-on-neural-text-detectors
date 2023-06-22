import os
from style_diff_test.anova_detectors import AnovaDetectors
from detectors.turnitin_detector import TurnitIn
from detectors.open_ai_classifier import OpenAiClassifier
from detectors.gpt_2_detector import GPT2Detector
from generators.gpt3_5_turbo import ChatGPT
import csv

standard_dir = "C:\\Users\\vital\\PycharmProjects\\M12Project\\char_mutation\\texts\\standard"
latin_a_dir = "C:\\Users\\vital\\PycharmProjects\\M12Project\\char_mutation\\texts\\latin_a"
latin_e_dir = "C:\\Users\\vital\\PycharmProjects\\M12Project\\char_mutation\\texts\\latin_e"
replace_l_dir = "C:\\Users\\vital\\PycharmProjects\\M12Project\\char_mutation\\texts\\replace_L"
results_dir = "C:\\Users\\vital\\PycharmProjects\\M12Project\\char_mutation\\results"


# Transformation functions
def replace_latin_cyrilic_a(text):
    cyrilic_char = u'\u0430'
    return text.replace("a", cyrilic_char) # replace latin o with cyrilic o


def replace_latin_cyrilic_e(text):
    cyrilic_char = u"\u0435"
    return text.replace("e", cyrilic_char) # replace latin e with cyrilic e


def replace_latin_il(text):
    return text.replace("l", "I")


transformations = [replace_latin_cyrilic_a, replace_latin_cyrilic_e, replace_latin_il]
transformed_directories = [latin_a_dir, latin_e_dir, replace_l_dir]

filepaths_standard = []
filepaths_latin_a = []
filepaths_latin_e = []
filepaths_replace_l = []

for i in range(10):
    filename = f'prompt_{i}.txt'
    filepaths_standard.append(os.path.join(standard_dir, filename))

    # Create transformed versions of the files and save them in the respective directories
    for transformation, directory in zip(transformations, transformed_directories):
        with open(filepaths_standard[-1], 'r', encoding='utf-8') as file:
            text = file.read()

        transformed_text = transformation(text)

        transformed_filepath = os.path.join(directory, filename)
        with open(transformed_filepath, 'w', encoding='utf-8') as file:
            file.write(transformed_text)

        # Append to corresponding list
        if directory == latin_a_dir:
            filepaths_latin_a.append(transformed_filepath)
        elif directory == latin_e_dir:
            filepaths_latin_e.append(transformed_filepath)
        elif directory == replace_l_dir:
            filepaths_replace_l.append(transformed_filepath)

# Rest of your code here...

detectoring = AnovaDetectors(turnitin=0, openai_classifier=0, generator=ChatGPT(),
                           gpt2_detector=GPT2Detector())

detectors = ['openai', 'gpt2', 'turnitin']

for detector in detectors:
    detector_csv_path = os.path.join(results_dir, f"{detector}_scores.csv")

    # Skip the detector if the corresponding CSV file already exists.
    if os.path.exists(detector_csv_path):
        print(f"Skipping {detector} analysis as its CSV file already exists.")
        continue

    print(f"{detector.capitalize()} analysis")

    if detector == "openai":
        detectoring.openai_classifier = OpenAiClassifier()
        detector_data = {
            'standard': detectoring.calculate_scores_openai(filepaths_standard),
            'latin_a': detectoring.calculate_scores_openai(filepaths_latin_a),
            'latin_e': detectoring.calculate_scores_openai(filepaths_latin_e),
            'latin_replace_l': detectoring.calculate_scores_openai(filepaths_replace_l)
        }
        detectoring.openai_classifier.close()

    elif detector == "gpt2":
        detector_data = {
            'standard': detectoring.calculate_scores_gpt2(filepaths_standard),
            'latin_a': detectoring.calculate_scores_gpt2(filepaths_latin_a),
            'latin_e': detectoring.calculate_scores_gpt2(filepaths_latin_e),
            'latin_replace_l': detectoring.calculate_scores_gpt2(filepaths_replace_l)
        }

    elif detector == "turnitin":
        detectoring.turnitin = TurnitIn()
        single_wait_time = 15
        wait_time = 60
        detector_data = {
            'standard': detectoring.calculate_scores_turnitin_no_regeneration(filepaths_standard),
            'latin_replace_l': detectoring.calculate_scores_turnitin_no_regeneration(filepaths_replace_l)
        }

        detectoring.turnitin.close()

    # Save detector's scores to CSVs
    with open(detector_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in detector_data.items():
            writer.writerow([key] + value)
        print(f"Successfully wrote {detector} scores to {detector_csv_path}")

