from typing import List
from detectors.turnitin_detector import TurnitIn
from generators.gpt3_5_turbo import ChatGPT
from parameter_optimizer import ParameterOptimizer
from detectors.open_ai_classifier import OpenAiClassifier
from detectors.gpt_2_detector import GPT2Detector
import os
import pickle
import threading

base_dir = "C:/Users/vital/PycharmProjects/M12Project/parameter_optimization"
param_names: List[str] = ['frequency_penalty', 'presence_penalty', 'temperature', "top_p"]
param_plot_names = ['Frequency penalty', 'Presence penalty', 'Temperature', 'Top P']
# parameters data note: bounds for frequency penalty is set to 1, not 2, because after 1 the
# model becomes too unpredictable. The same is true for temperature
# findings: presence penalty - ok, p - ok. frequency and temp should be done carefully, makes the model unpredictable.
values = {
        'frequency_penalty': {'fixed_value': 0, 'bounds': [0, 1], 'step': 0.1},
        'presence_penalty': {'fixed_value': 0, 'bounds': [0, 2], 'step': 0.1},
        'temperature': {'fixed_value': 1, 'bounds': [0, 1], 'step': 0.1},
        'top_p': {'fixed_value': 1, 'bounds': [0, 1], 'step': 0.1}
        }
tokens = 800
dir_paths = {}

for param in param_names:
    dir_paths[param] = {
        'results': os.path.join(base_dir, param, 'results'),
        'texts': os.path.join(base_dir, param, 'texts'),
        'logs': os.path.join(base_dir, param, 'logs')
    }


def load_prompts(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return content.split('\n\n')

train_prompts = [
    "Write a formal five-hundred-word argumentative essay on the topic 'Should school students be allowed to curate their high school curriculum?'",
    "Write a formal five-hundred-word argumentative essay on the topic 'Should the death sentence be implemented globally?'",
    "Write a formal five-hundred-word argumentative essay on the topic 'Should the government do more to improve accessibility for people with physical disabilities?'",
]

generator = ChatGPT()

def generate():
    final = {}

    threads = []
    for current_param in param_names:
        optimizer = ParameterOptimizer(generator=generator,
                                       turnitin=0,
                                       openai_classifier=0,
                                       gpt2_detector=0,
                                       max_tokens=tokens,
                                       param_name=current_param,
                                       param_bounds=values[current_param]['bounds'],
                                       param_step=values[current_param]['step'],
                                       texts_dir_path=dir_paths[current_param]['texts'],
                                       result_dir_path=dir_paths[current_param]['results'],
                                       log_dir_path=dir_paths[current_param]['logs'],
                                       train_prompts=train_prompts
                                       )

        # Start a new thread for each optimizer
        thread = threading.Thread(target=lambda: final.update({current_param: optimizer.generate_and_save_texts()}))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    return final


def analyze_initial_detectors(final):

    for current_param, plot_name in zip(param_names, param_plot_names):
        optimizer = ParameterOptimizer(generator=generator,
                                       turnitin=0,
                                       openai_classifier=0,
                                       gpt2_detector=0,
                                       max_tokens=tokens,
                                       param_name=current_param,
                                       param_bounds=values[current_param]['bounds'],
                                       param_step=values[current_param]['step'],
                                       texts_dir_path=dir_paths[current_param]['texts'],
                                       result_dir_path=dir_paths[current_param]['results'],
                                       log_dir_path=dir_paths[current_param]['logs'],
                                       train_prompts=train_prompts
                                       )
        file_paths_dict = final[current_param]

        # OpenAI classifier
        # Check for existing OpenAI pickle data
        if os.path.exists(f'{current_param}_openai_classifier.pickle'):
            with open(f'{current_param}_openai_classifier.pickle', 'rb') as f:
                openai_data = pickle.load(f)
            scores_openai_dict = openai_data
        else:
            openai = OpenAiClassifier()
            optimizer.openai_classifier = openai
            scores_openai_dict = optimizer.submit_and_calculate_scores_openai(file_paths_dict)
            optimizer.save_scores_to_csv(scores_openai_dict, 'OpenAI classifier', plot_name)
            # Save the results of OpenAI to pickle
            with open(f'{current_param}_openai_classifier.pickle', 'wb') as f:
                pickle.dump(scores_openai_dict, f)
            openai.close()

        # GPT-2
        # Check for existing GPT2 pickle data
        if os.path.exists(f'{current_param}_gpt2_detector.pickle'):
            with open(f'{current_param}_gpt2_detector.pickle', 'rb') as f:
                gpt2_data = pickle.load(f)
            scores_gpt2_dict = gpt2_data
        else:
            gpt2 = GPT2Detector()
            optimizer.gpt2_detector = gpt2
            scores_gpt2_dict = optimizer.submit_and_calculate_scores_gpt2(file_paths_dict)
            optimizer.save_scores_to_csv(scores_gpt2_dict, 'GPT2 detector', plot_name)
            # Save the results of GPT2 to pickle
            with open(f'{current_param}_gpt2_detector.pickle', 'wb') as f:
                pickle.dump(scores_gpt2_dict, f)



def analyze_turnitin_detector(final):

    for current_param, plot_name in zip(param_names, param_plot_names):
        optimizer = ParameterOptimizer(generator=generator,
                                       turnitin=0,
                                       openai_classifier=0,
                                       gpt2_detector=0,
                                       max_tokens=tokens,
                                       param_name=current_param,
                                       param_bounds=values[current_param]['bounds'],
                                       param_step=values[current_param]['step'],
                                       texts_dir_path=dir_paths[current_param]['texts'],
                                       result_dir_path=dir_paths[current_param]['results'],
                                       log_dir_path=dir_paths[current_param]['logs'],
                                       train_prompts=train_prompts
                                       )
        file_paths_dict = final[current_param]

        # Load Turnitin data from pickle if it exists, otherwise calculate
        if os.path.exists(f'{current_param}_turnitin_detector.pickle'):
            with open(f'{current_param}_turnitin_detector.pickle', 'rb') as f:
                turnitin_data = pickle.load(f)
            scores_turnitin_dict = turnitin_data['Turnitin']
            null_counts_dict = turnitin_data['Nulls']
        else:
            # Third detector: Turnitin
            turnitin = TurnitIn()
            optimizer.turnitin = turnitin
            scores_turnitin_dict, null_counts_dict = optimizer.submit_and_calculate_scores_turnitin(file_paths_dict,
                                                                                                    train_prompts, 50,
                                                                                                    15)
            # Save the results of Turnitin to pickle
            with open(f'{current_param}_turnitin_detector.pickle', 'wb') as f:
                pickle.dump({'Turnitin': scores_turnitin_dict, 'Nulls': null_counts_dict}, f)
            turnitin.close()

        optimizer.save_scores_to_csv(scores_turnitin_dict, 'Turnitin', plot_name)
        optimizer.save_null_counts_to_csv(null_counts_dict, current_param)


def combine_results():

    for current_param, plot_name in zip(param_names, param_plot_names):
        optimizer = ParameterOptimizer(generator=generator,
                                       turnitin=0,
                                       openai_classifier=0,
                                       gpt2_detector=0,
                                       max_tokens=tokens,
                                       param_name=current_param,
                                       param_bounds=values[current_param]['bounds'],
                                       param_step=values[current_param]['step'],
                                       texts_dir_path=dir_paths[current_param]['texts'],
                                       result_dir_path=dir_paths[current_param]['results'],
                                       log_dir_path=dir_paths[current_param]['logs'],
                                       train_prompts=train_prompts
                                       )
        scores_turnitin_dict = optimizer.load_scores_from_csv('Turnitin', plot_name)
        scores_openai_dict = optimizer.load_scores_from_csv('OpenAI classifier', plot_name)
        scores_gpt2_dict = optimizer.load_scores_from_csv('GPT2 detector', plot_name)

        # total result
        all_detectors = ['Turnitin', 'OpenAI classifier', 'GPT2 detector']
        all_scores_dicts = [scores_turnitin_dict, scores_openai_dict, scores_gpt2_dict]
        optimizer.save_scores_to_combined_csv(all_scores_dicts, all_detectors, plot_name)
        optimizer.plot_combined_scores(all_scores_dicts, all_detectors, plot_name)

# generating and saving texts
# Check if the pickle file already exists
pickle_file = "filepath_dict.pickle"
# os.remove(pickle_file)
if os.path.exists(pickle_file):
    # Load the file paths from the pickle file
    with open(pickle_file, "rb") as file:
        final_dict = pickle.load(file)
        print(final_dict)
else:
    final_dict = generate()
    # Save the dictionary to a pickle file
    with open(pickle_file, "wb") as file:
        pickle.dump(final_dict, file)


analyze_turnitin_detector(final_dict)
analyze_initial_detectors(final_dict)
combine_results()
print("OMG DONE")