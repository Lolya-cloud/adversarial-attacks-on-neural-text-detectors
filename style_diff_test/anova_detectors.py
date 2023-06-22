import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from generators.gpt3_5_turbo import ChatGPT
from generators.openAiParameters import Parameters
from detectors.turnitin_detector import TurnitIn
from detectors.open_ai_classifier import OpenAiClassifier
from utilities.TXTHandler import TXTHandler, TXTLoader
import logging
import ast
import csv


class AnovaDetectors:
    def __init__(self, generator, turnitin, gpt2_detector, openai_classifier):
        self.generator = generator
        self.turnitin = turnitin
        self.gpt2_detector = gpt2_detector
        self.openai_classifier = openai_classifier
        self.file_paths_dict = {}

    # parameter = [filepath1, filepath2, ...,]
    def calculate_scores_gpt2(self, filepaths_list):
        scores = []
        for filepath in filepaths_list:
            text = TXTLoader.load_file(filepath)
            scores.append(self.gpt2_detector.submit_and_scrape(text).fake_probability)
        return scores

    def calculate_scores_openai(self, filepaths_list):
        labels = []
        for filepath in filepaths_list:
            text = TXTLoader.load_file(filepath)
            label = self.openai_classifier.submit_and_scrape_label(text)
            print(f"Label for {filepath}: {label}")
            labels.append(label)
        return labels

    def calculate_scores_turnitin(self, filepaths_list, prompt_list, parameters, wait_time, single_wait_time):
        scores = self.turnitin.submit_and_scrape_existing_files_list(filepaths_list, wait_time=wait_time)
        regeneration_count = 0  # Counter for the total number of texts regenerated

        # Iterate until there are no None values in the scores
        while None in scores:
            # Generate new text for scores that are None and update the score
            for i, score in enumerate(scores):
                if score is None:
                    # Extract information from file path
                    dir_name, file_name = os.path.split(filepaths_list[i])
                    prompt = prompt_list[i]

                    # Generate new text
                    print(f"None encountered, regenerating text: {prompt}, {dir_name}, {file_name}")
                    self.generator.generate_prompt_save_txt(prompt, parameters, dir_name, file_name)
                    regeneration_count += 1

                    # Update the score for this text
                    new_score = self.turnitin.submit_and_scrape_existing_files(filepaths_list[i],
                                                                              wait_time=single_wait_time)
                    scores[i] = new_score

        return scores, regeneration_count

    def calculate_scores_turnitin_no_regeneration(self, filepaths_list, wait_time=45, single_wait_time=15):
        scores = self.turnitin.submit_and_scrape_existing_files_list(filepaths_list, wait_time=wait_time)
        while None in scores:
            # Generate new text for scores that are None and update the score
            for i, score in enumerate(scores):
                if score is None:
                    # Update the score for this text
                    print("None score, reuploading")
                    new_score = self.turnitin.submit_and_scrape_existing_files(filepaths_list[i],
                                                                               wait_time=single_wait_time)
                    scores[i] = new_score

        return scores

    def calculate_scores_openai_num(self, filepaths_list):
        scores = []
        for filepath in filepaths_list:
            text = TXTLoader.load_file(filepath)
            score = self.openai_classifier.submit_and_scrape(text)
            print(f"Score for {filepath}: {score}")
            scores.append(score)
        return scores


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


def load_scores_turnitin(detector_name, directory):
    scores_dict = {}
    file_path = os.path.join(directory, f'{detector_name}_results.csv')
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader) # skip header
        for row in reader:
            label = row[0]
            # Use ast.literal_eval to convert string to tuple
            scores_tuple = ast.literal_eval(row[1])
            # Get the first element of the tuple (i.e., the list of scores)
            scores = scores_tuple[0]
            scores_dict[label] = scores
    return scores_dict
