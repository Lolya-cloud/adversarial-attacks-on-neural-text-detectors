from generators.gpt3_5_turbo import ChatGPT
from generators.openAiParameters import Parameters
from detectors.turnitin_detector import TurnitIn
from detectors.gpt_2_detector import GPT2Detector, GPT2Output
from detectors.open_ai_classifier import OpenAiClassifier
from utilities.TXTHandler import TXTHandler, TXTLoader
import pandas as pd
import os


class Optimizer():
    def __init__(self, texts_dir, res_dir, generator):
        self.texts_dir = texts_dir
        self.res_dir = res_dir
        self.generator = generator

    def generate_save_text(self, prompt, parameters, filename):
        # generates a single prompt with the given parameters
        filepath = self.generator.generate_prompt_save_txt_bounds(prompt, parameters, self.texts_dir, filename, 425, 575)
        return filepath

    def generate_many_save_txt(self, prompt, prompt_num):
        # Create a DataFrame to store the parameters and corresponding file paths
        df = pd.DataFrame(columns=['frequency_penalty', 'presence_penalty', 'filepath'])

        # Define the ranges for frequency_penalty and presence_penalty
        frequency_penalties = [round(i * 0.1, 1) for i in range(7)]  # 0.0, 0.1, ..., 0.6
        presence_penalties = [round(i * 0.1, 1) for i in range(6)]  # 0.0, 0.1, ..., 0.5

        # Iterate over all combinations of parameters
        for frequency_penalty in frequency_penalties:
            for presence_penalty in presence_penalties:
                # Define the parameters
                parameters = Parameters(max_tokens=800, frequency_penalty=frequency_penalty,
                                        presence_penalty=presence_penalty)

                # Define the filename based on the parameters
                filename = f"fp_{frequency_penalty}_pp_{presence_penalty}_prompt_num.txt"

                # Generate the text and get the filepath
                filepath = self.generate_save_text(prompt, parameters, filename)
                print(f"Generated text for prompt_num: {prompt_num}, fp: {frequency_penalty}, pp: {presence_penalty}"
                      f", with filepath: {filepath}")
                # Append the parameters and filepath to the DataFrame
                df = df.append({'frequency_penalty': frequency_penalty, 'presence_penalty': presence_penalty,
                                'filepath': filepath}, ignore_index=True)

        # Return the DataFrame
        return df

    # parameter = [filepath1, filepath2, ...,]
    def calculate_scores_gpt2(self, filepaths_list):
        scores = []
        gpt2_detector = GPT2Detector()
        for filepath in filepaths_list:
            text = TXTLoader.load_file(filepath)
            score = gpt2_detector.submit_and_scrape(text).fake_probability
            scores.append(score)
            print(f"GPT2 output score for filepath: {filepath}, score: {score}")
        return scores

    # Returns original label produced by the classifier.
    # For numerical mapping analogue check calculate_scores_openai_num method
    def calculate_scores_openai(self, filepaths_list):
        labels = []
        openai_classifier = OpenAiClassifier()
        for filepath in filepaths_list:
            text = TXTLoader.load_file(filepath)
            label = openai_classifier.submit_and_scrape_label(text)
            print(f"Label for {filepath}: {label}")
            labels.append(label)
        return labels

    # returns numerical value accordingly to the official open ai mapping of their result labels
    def calculate_scores_openai_num(self, filepaths_list):
        scores = []
        openai_classifier = OpenAiClassifier()
        for filepath in filepaths_list:
            text = TXTLoader.load_file(filepath)
            score = openai_classifier.submit_and_scrape(text)
            print(f"Score for {filepath}: {score}")
            scores.append(score)
        return scores

    def calculate_scores_turnitin(self, filepaths_list, prompt_list, parameters, wait_time, single_wait_time):
        turnitin = TurnitIn()
        scores = turnitin.submit_and_scrape_existing_files_list(filepaths_list, wait_time=wait_time)
        regeneration_count = 0  # Counter for the total number of texts regenerated

        # Iterate until there are no None values in the scores
        while None in scores:
            # Generate new text for scores that are None and update the score
            for i, score in enumerate(scores):
                if score is None:
                    # Extract information from file path
                    dir_name, file_name = os.path.split(filepaths_list[i])
                    prompt = prompt_list[i]
                    parameter = parameters[i]

                    # Generate new text
                    print(f"None encountered, regenerating text: {prompt}, {dir_name}, {file_name}, frequency_penalty: "
                          f"{parameter.frequency_penalty}, presence_penalty: {parameter.presence_penalty}")
                    self.generator.generate_prompt_save_txt(prompt, parameter, dir_name, file_name)
                    regeneration_count += 1

                    # Update the score for this text
                    new_score = turnitin.submit_and_scrape_existing_files(filepaths_list[i],
                                                                               wait_time=single_wait_time)
                    scores[i] = new_score

        return scores, regeneration_count

    def analyse_turnitin(self, dataframe, prompt):
        # Get filepaths and Parameters from the dataframe
        filepaths_list = dataframe['filepath'].tolist()
        parameters_list = [Parameters(max_tokens=800, frequency_penalty=row['frequency_penalty'],
                                      presence_penalty=row['presence_penalty']) for index, row in dataframe.iterrows()]

        # Prepare a list of the same prompt repeated for each filepath
        prompt_list = [prompt] * len(filepaths_list)

        # Calculate scores using Turnitin detector
        scores, null_counts = self.calculate_scores_turnitin(filepaths_list, prompt_list, parameters_list, 40, 15)
        print(f"Number of regenerated texts for Turnitin: {null_counts}")

        # Add Turnitin scores to the dataframe
        dataframe['turnitin_scores'] = scores
        filepath = self.save_dataframe(dataframe)
        return filepath

    def analyse_openai(self, dataframe):
        # Get filepaths from the dataframe
        filepaths_list = dataframe['filepath'].tolist()

        # Calculate scores using OpenAI classifier
        scores = self.calculate_scores_openai_num(filepaths_list)

        # Add OpenAI scores to the dataframe
        dataframe['openai_scores'] = scores
        filepath = self.save_dataframe(dataframe)
        return filepath

    def analyse_gpt2(self, dataframe):
        # Get filepaths from the dataframe
        filepaths_list = dataframe['filepath'].tolist()

        # Calculate scores using GPT-2 detector
        scores = self.calculate_scores_gpt2(filepaths_list)

        # Add GPT-2 scores to the dataframe
        dataframe['gpt2_scores'] = scores
        filepath = self.save_dataframe(dataframe)
        return filepath

    def save_dataframe(self, dataframe):
        filepath = os.path.join(self.res_dir, 'dataframe.csv')
        dataframe.to_csv(filepath, index=False)
        return filepath

    def load_dataframe(self):
        filepath = os.path.join(self.res_dir, 'dataframe.csv')
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            return None
