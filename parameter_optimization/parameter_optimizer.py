import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from generators.gpt3_5_turbo import ChatGPT
from generators.openAiParameters import Parameters
from detectors.turnitin_detector import TurnitIn
from detectors.open_ai_classifier import OpenAiClassifier
import logging


class ParameterOptimizer:
    def __init__(self, generator, turnitin, gpt2_detector, openai_classifier, max_tokens, param_name, param_bounds, param_step, texts_dir_path, result_dir_path,
                 log_dir_path, train_prompts):
        self.generator = generator
        self.turnitin = turnitin
        self.gpt2_detector = gpt2_detector
        self.openai_classifier = openai_classifier
        self.max_tokens = max_tokens
        self.param_name = param_name
        self.param_bounds = param_bounds
        self.param_step = param_step
        self.texts_dir_path = texts_dir_path
        self.result_dir_path = result_dir_path
        self.log_dir_path = log_dir_path
        self.train_prompts = train_prompts
        self.file_paths_dict = {}
        logging.basicConfig(filename=os.path.join(self.log_dir_path, 'parameter_optimization.log'), level=logging.INFO,
                            format='%(asctime)s %(message)s')

    def generate_and_save_texts(self):
        param_values = np.arange(self.param_bounds[0], self.param_bounds[1] + self.param_step, self.param_step)
        for param_value in param_values:
            param_value = round(param_value.item(), 2)
            parameters = Parameters(max_tokens=self.max_tokens)
            parameters.__dict__[self.param_name] = param_value
            print(parameters.__dict__)
            logging.info(parameters.__dict__)

            for idx, prompt in enumerate(self.train_prompts):
                file_name = f"{self.param_name}_{param_value}_prompt_{idx}.txt"
                self.generator.generate_prompt_save_txt(prompt, parameters, self.texts_dir_path, file_name)
                logging.info(prompt)
                file_path = os.path.join(self.texts_dir_path, file_name)
                # Add the file path to the dictionary
                self.file_paths_dict.setdefault(f"{self.param_name}_{param_value}", []).append(file_path)

        # Return file paths
        return self.file_paths_dict

    def submit_and_calculate_scores_turnitin(self, file_paths_dict, prompts_list, waittime, single_file_waittime):
        average_scores_dict = {}
        null_counts_dict = {}  # New dictionary to count null values
        null_present = True  # to keep the loop going until no nulls found

        while null_present:
            null_present = False  # reset for each complete run
            scores_dict = self.turnitin.submit_and_scrape_existing_files(file_paths_dict, waittime)
            # Creating a temporary dictionary to hold file paths with None scores
            file_paths_dict_temp = {}
            for param_value, scores in scores_dict.items():
                logging.info(f"Scores for {self.param_name} = {param_value}: {scores}")
                new_scores = [round(score / 100, 2) if score is not None else None for score in scores]

                if param_value not in null_counts_dict:
                    null_counts_dict[param_value] = 0  # Initialize count if not exist

                # Regenerate text for null scores and re-check
                for i, score in enumerate(new_scores):
                    if score is None:
                        null_present = True  # null found, another run required
                        null_counts_dict[param_value] += 1  # Increase the count
                        file_path = file_paths_dict[param_value][i]
                        directory, filename = os.path.split(file_path)
                        prompt_index = int(filename.split('_')[-1].split('.')[0])  # Extract prompt index from filename
                        prompt = prompts_list[prompt_index]  # Retrieve the prompt from the prompt list
                        print(f"Null text: {file_path}, Score value: {score}, Prompt index: {prompt_index}")

                        # Prepare parameters
                        parameters = Parameters(max_tokens=self.max_tokens)
                        parameters.__dict__[self.param_name] = float(param_value.split('_')[-1])

                        # Generate new texts
                        print(
                            f"generating new text for {filename} with filepath={file_path} and parameters: {parameters}")
                        self.generator.generate_prompt_save_txt(prompt, parameters, directory, filename)

                        # Add the regenerated file's path to the temporary dictionary
                        if param_value not in file_paths_dict_temp:
                            file_paths_dict_temp[param_value] = []
                        file_paths_dict_temp[param_value].append(file_path)

                        # Check the scores again
                        single_file_dict = {param_value: [file_path]}  # Adjust the dict for the specific file
                        new_score_dict = self.turnitin.submit_and_scrape_existing_files(single_file_dict,
                                                                                        single_file_waittime)
                        new_scores[i] = new_score_dict[param_value][0]

                total = sum(score for score in new_scores if score is not None)
                average_scores_dict[param_value] = total / len(new_scores)
            # Update file_paths_dict to include only file paths which had None scores in the current iteration
            file_paths_dict = file_paths_dict_temp

        return average_scores_dict, null_counts_dict

    def submit_and_calculate_scores_openai(self, file_paths_dict):
        scores_dict = {}
        for param, file_paths in file_paths_dict.items():
            scores_list = []
            for file_path in file_paths:
                print(f"OpenAI: {param}: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as file:  # Open each file
                    file_content = file.read()  # Read the content of the file
                    score = self.openai_classifier.submit_and_scrape(file_content)  # Execute the submit_and_scrape function
                    scores_list.append(score)  # Append the score to the scores list
            new_scores = [round(score, 2) for score in scores_list]
            scores_dict[param] = sum(new_scores) / len(scores_list)  # Assign the scores list to the corresponding key in the dictionary
        return scores_dict

    def submit_and_calculate_scores_gpt2(self, file_paths_dict):
        scores_dict = {}
        for param, file_paths in file_paths_dict.items():
            scores_list = []
            for file_path in file_paths:
                print(f"GPT2: {param}: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as file:  # Open each file
                    file_content = file.read()  # Read the content of the file
                    score = self.gpt2_detector.submit_and_scrape(
                        file_content).fake_probability  # Execute the submit_and_scrape function
                    scores_list.append(score)  # Append the score to the scores list
            new_scores = [round(score, 2) for score in scores_list]
            scores_dict[param] = sum(new_scores) / len(scores_list)
        return scores_dict

    def save_scores_to_csv(self, scores_dict, detector_name, param_name):
        new_scores_dict = {float(key.split('_')[-1]): value for key, value in scores_dict.items()}
        df = pd.DataFrame(list(new_scores_dict.items()), columns=[param_name, 'Score'])
        df.to_csv(f'{self.result_dir_path}/{self.param_name}_{detector_name}_scores.csv', index=False)

    def load_scores_from_csv(self, detector_name, param_name):
        # Define the path to the csv file
        file_path = f'{self.result_dir_path}/{self.param_name}_{detector_name}_scores.csv'

        # Read the csv file into a DataFrame
        df = pd.read_csv(file_path)

        # Convert the DataFrame back into a dictionary
        scores_dict = {f"{self.param_name}_{row[param_name]}": row['Score'] for index, row in df.iterrows()}

        return scores_dict

    def plot_scores(self, scores_dict, detector_name, param_name):
        new_scores_dict = {float(key.split('_')[-1]): value for key, value in scores_dict.items()}
        plt.plot(list(new_scores_dict.keys()), list(new_scores_dict.values()), label=f'{detector_name} score')
        plt.legend()
        plt.xlabel(param_name)
        plt.ylabel(f'{detector_name} score')
        plt.title('Parameter tuning results')
        plt.savefig(f'{self.result_dir_path}/{self.param_name}_{detector_name}_scores.png')
        plt.show()

    def save_scores_to_combined_csv(self, scores_dicts, detector_names, param_name):
        df_combined = None
        for scores_dict, detector_name in zip(scores_dicts, detector_names):
            new_scores_dict = {float(key.split('_')[-1]): value for key, value in scores_dict.items()}
            df = pd.DataFrame(list(new_scores_dict.items()), columns=[param_name, detector_name])
            if df_combined is None:
                df_combined = df
            else:
                df_combined = df_combined.merge(df, on=param_name)
        df_combined.to_csv(f'{self.result_dir_path}/{self.param_name}_combined_scores.csv', index=False)

    def plot_combined_scores(self, scores_dicts, detector_names, param_name):
        plt.figure(figsize=(5, 4))
        for scores_dict, detector_name in zip(scores_dicts, detector_names):
            new_scores_dict = {float(key.split('_')[-1]): value for key, value in scores_dict.items()}
            plt.plot(list(new_scores_dict.keys()), list(new_scores_dict.values()), label=f'{detector_name} score')
        plt.legend()
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.title('Parameter tuning results')
        plt.savefig(f'{self.result_dir_path}/{self.param_name}_combined_scores.png')
        plt.show()

    def save_null_counts_to_csv(self, null_counts_dict, param_name):
        df = pd.DataFrame(list(null_counts_dict.items()), columns=[param_name, 'Null Counts'])
        df.to_csv(f'{self.result_dir_path}/{self.param_name}_null_counts.csv', index=False)

    def plot_null_counts(self, null_counts_dict, param_name):
        plt.bar(list(null_counts_dict.keys()), list(null_counts_dict.values()), label='Null counts')
        plt.xlabel(param_name)
        plt.ylabel('Null counts')
        plt.title('Number of documents unclassified by Turnitin')
        plt.savefig(f'{self.result_dir_path}/{self.param_name}_null_counts.png')
        plt.show()
