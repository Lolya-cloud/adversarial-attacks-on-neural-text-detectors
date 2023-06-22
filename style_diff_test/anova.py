from detectors.turnitin_detector import TurnitIn
from detectors.gpt_2_detector import GPT2Detector
from detectors.open_ai_classifier import OpenAiClassifier
from generators.gpt3_5_turbo import ChatGPT
from generators.openAiParameters import Parameters
from anova_detectors import AnovaDetectors
import os
import pickle
import threading
import pandas as pd
import scipy.stats as stats
import ast

# load prompts
prompts_dir = "C:/Users/vital/PycharmProjects/M12Project/prompts/essay_style_prompts_500"
texts_dir = "C:/Users/vital/PycharmProjects/M12Project/style_diff_test/texts"
results_directory = "C:/Users/vital/PycharmProjects/M12Project/style_diff_test/results"
essay_formats = [
    "argumentative",
    "cause_and_effect",
    "compare_contrast",
    "controversial_argumentative",
    "descriptive",
    "expository",
    "funny_argumentative",
    "narrative",
    "persuasive",
    "research"]

prompts = {}
prompts_per_format = 20


def load_prompts(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return content.split('\n\n')


# prompt files. each file has 20 prompts for a specific style.
for j in essay_formats:
    prompt_list = load_prompts(os.path.join(prompts_dir, f"{j}.txt"))
    prompts[j] = prompt_list


def check_prompts():
    # Check if all essay_formats are in prompts
    for essay_format in essay_formats:
        if essay_format not in prompts:
            print(f"Error: {essay_format} is not in prompts!")
            break
        elif len(prompts[essay_format]) != 20:
            print(f"Error: {essay_format} does not have exactly 20 prompts!")
            break
    else:  # If loop completes without break
        print("All essay formats are in prompts and each has exactly 20 prompts!")

    # Print out dictionary in blocks
    for key, value in prompts.items():
        print(f"{key}:")
        for prompt in value:
            print(prompt)
        print("\n")


# check_prompts()


def generate_texts():
    generator = ChatGPT()
    parameters = Parameters(max_tokens=800)
    print(parameters.__dict__)
    paths = {}

    def handle_prompt(essay_style):
        txt_files = []
        for i, prompt in enumerate(prompts[essay_style]):
            file_name = f"prompt_{i}.txt"
            dir_name = os.path.join(texts_dir, essay_style)
            # Call generate_prompt_save_txt() method
            print(f"Generating: prompt: {prompt}; dir_name: {dir_name}; file_name: {file_name}")
            generator.generate_prompt_save_txt(prompt, parameters, dir_name, file_name)
            # Store file path
            txt_files.append(os.path.join(dir_name, file_name))
        return {essay_style: txt_files}

    # Create new thread for each key in the dictionary
    threads = []
    for key in prompts:
        thread = threading.Thread(target=lambda k=key: paths.update(handle_prompt(k)))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All threads completed.")
    return paths


def analyze_gpt2(texts_paths):
    detector_name = 'GPT2 output classifier'
    csv_file_path = os.path.join(results_directory, f'{detector_name}_results.csv')

    if os.path.exists(csv_file_path):
        # If the CSV file exists, load the results from the CSV file
        gpt2_scores = load_detector_results_from_csv(detector_name, results_directory)
    else:
        # Otherwise, perform the analysis
        gpt2_scores = {}
        detectors = AnovaDetectors(generator=0, turnitin=0, gpt2_detector=GPT2Detector(), openai_classifier=0)
        for essay_format in essay_formats:
            style_paths = texts_paths[essay_format]
            scores = detectors.calculate_scores_gpt2(style_paths)
            gpt2_scores[essay_format] = scores

        # Save the results to a CSV file
        save_detector_results_to_csv(detector_name, gpt2_scores, results_directory)

    return gpt2_scores


def analyze_openai(texts_paths):
    detector_name = 'OpenaAI classifier'
    csv_file_path = os.path.join(results_directory, f'{detector_name}_results.csv')

    if os.path.exists(csv_file_path):
        # If the CSV file exists, load the results from the CSV file
        openai_scores = load_detector_results_from_csv(detector_name, results_directory)
    else:
        # Otherwise, perform the analysis
        openai_scores = {}
        detectors = AnovaDetectors(generator=0, turnitin=0, gpt2_detector=0, openai_classifier=OpenAiClassifier())
        for essay_format in essay_formats:
            style_paths = texts_paths[essay_format]
            scores = detectors.calculate_scores_openai(style_paths)
            openai_scores[essay_format] = scores
        detectors.openai_classifier.close()

        # Save the results to a CSV file
        save_detector_results_to_csv(detector_name, openai_scores, results_directory)

    return openai_scores


def analyze_turnitin(texts_paths):
    detector_name = 'Turnitin'
    csv_file_path = os.path.join(results_directory, f'{detector_name}_results.csv')

    if os.path.exists(csv_file_path):
        # If the CSV file exists, load the results from the CSV file
        turnitin_scores = load_detector_results_from_csv(detector_name, results_directory)
    else:
        # Otherwise, perform the analysis
        turnitin_scores = {}
        detectors = AnovaDetectors(generator=0, turnitin=TurnitIn(), gpt2_detector=0, openai_classifier=0)
        for essay_format in essay_formats:
            style_paths = texts_paths[essay_format]
            parameters = Parameters(max_tokens=800)
            scores = detectors.calculate_scores_turnitin(style_paths, prompts[essay_format], parameters, 30, 15)
            turnitin_scores[essay_format] = scores
        detectors.turnitin.close()

        # Save the results to a CSV file
        save_detector_results_to_csv(detector_name, turnitin_scores, results_directory)

    return turnitin_scores


def save_detector_results_to_csv(detector_name, results_dict, directory):
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(list(results_dict.items()), columns=['Essay Style', 'Scores'])

    # Define the file path
    file_path = os.path.join(directory, f'{detector_name}_results.csv')

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)


def load_detector_results_from_csv(detector_name, directory):
    # Define the file path
    file_path = os.path.join(directory, f'{detector_name}_results.csv')

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Convert the "Scores" column from strings to lists
    df['Scores'] = df['Scores'].apply(ast.literal_eval)

    # Convert the DataFrame back into a dictionary
    results_dict = df.set_index('Essay Style')['Scores'].to_dict()

    return results_dict

pickle_file = 'texts_paths.pickle'
# Check if pickle file exists
if not os.path.exists(pickle_file):
    # Execute the text generation code
    texts_paths = generate_texts()

    # Save texts_paths to a pickle file
    with open(pickle_file, 'wb') as f:
        pickle.dump(texts_paths, f)
else:
    # Load texts_paths from the pickle file
    with open(pickle_file, 'rb') as f:
        texts_paths = pickle.load(f)

print("Starting gpt2 analysis")
gpt2_scores = analyze_gpt2(texts_paths)
print(gpt2_scores)
print("Starting openai analysis")
openai_scores = analyze_openai(texts_paths)
print(openai_scores)
print("Starting turnitin analysis")
turnitin_scores = analyze_turnitin(texts_paths)
print(turnitin_scores)


