from generators.gpt3_5_turbo import ChatGPT
from generators.openAiParameters import Parameters
from detectors.open_ai_classifier import OpenAiClassifier
from detectors.turnitin_detector import TurnitIn
from detectors.gpt_2_detector import GPT2Detector
from style_diff_test.anova_detectors import AnovaDetectors
import os
import csv
import pickle
import threading

iterations = 10
generator = ChatGPT()

texts_dir_standard = "C:\\Users\\vital\\PycharmProjects\\M12Project\\advanced_prompt_engineering\\perplexity_burstiness\\texts\\standard"
texts_dir_explanation = "C:\\Users\\vital\\PycharmProjects\\M12Project\\advanced_prompt_engineering\\perplexity_burstiness\\texts\\explanation"
texts_dir_perplexity = "C:\\Users\\vital\\PycharmProjects\\M12Project\\advanced_prompt_engineering\\perplexity_burstiness\\texts\\perplexity"
texts_dir_burst = "C:\\Users\\vital\\PycharmProjects\\M12Project\\advanced_prompt_engineering\\perplexity_burstiness\\texts\\burstiness"
texts_dir_both = "C:\\Users\\vital\\PycharmProjects\\M12Project\\advanced_prompt_engineering\\perplexity_burstiness\\texts\\both"
texts_dir_regen_standard = "C:\\Users\\vital\\PycharmProjects\\M12Project\\advanced_prompt_engineering\\perplexity_burstiness\\texts\\reg_standard"
texts_dir_regen_explanation = "C:\\Users\\vital\\PycharmProjects\\M12Project\\advanced_prompt_engineering\\perplexity_burstiness\\texts\\reg_explanation"
texts_dir_regen_perplexity = "C:\\Users\\vital\\PycharmProjects\\M12Project\\advanced_prompt_engineering\\perplexity_burstiness\\texts\\reg_perplexity"
texts_dir_regen_burst = "C:\\Users\\vital\\PycharmProjects\\M12Project\\advanced_prompt_engineering\\perplexity_burstiness\\texts\\reg_burst"
texts_dir_regen_both = "C:\\Users\\vital\\PycharmProjects\\M12Project\\advanced_prompt_engineering\\perplexity_burstiness\\texts\\reg_both"
res_dir = "C:\\Users\\vital\\PycharmProjects\\M12Project\\advanced_prompt_engineering\\perplexity_burstiness\\results"

# Empty lists for filepaths
filepaths_standard = []
filepaths_explanation = []
filepaths_perplexity = []
filepaths_burstiness = []
filepaths_both = []
filepaths_regen_standard = []
filepaths_regen_explanation = []
filepaths_regen_perplexity = []
filepaths_regen_burst = []
filepaths_regen_both = []

# Empty lists for prompts
prompts_standard = []
prompts_explanation = []
prompts_perplexity = []
prompts_burstiness = []
prompts_both = []
prompts_regen_standard = []
prompts_regen_explanation = []
prompts_regen_perplexity = []
prompts_regen_burst = []
prompts_regen_both = []

# Filling the prompt variables
prompt_standard = "Write a five-hundred-words(500) essay on the topic 'Should school students be allowed to curate their high school curriculum?'"
prompt_explanation = "Write a five-hundred-words(500) essay on the topic 'Should school students be allowed to curate their high school curriculum?'. When it comes to writing content, two factors are crucial, perplexity and burstiness. Perplexity measures the complexity of the text. Separately, burstiness compares the variations of sentences. Humans tend to write with greater burstiness, for example, with some longer or more complex sentences alongside shorter ones. AI sentences tend to be more uniform. Therefore, when writing the following content, I need it to have a good amount of perplexity and burstiness."
prompt_perplexity = "Write a five-hundred-words(500) essay on the topic 'Should school students be allowed to curate their high school curriculum?'. Maximize the perplexity of the text."
prompt_burstiness = "Write a five-hundred-words(500) essay on the topic 'Should school students be allowed to curate their high school curriculum?'. Maximize the burstiness of the text."
prompt_both = "Write a five-hundred-words(500) essay on the topic 'Should school students be allowed to curate their high school curriculum?'. Maximize the perplexity and burstiness of the text."
prompt_regen_standard = "Rewrite the above essay in order to avoid AI detection."
prompt_regen_explanation = "Rewrite the above essay utilizing the following concepts: When it comes to writing content, two factors are crucial, perplexity and burstiness. Perplexity measures the complexity of the text. Separately, burstiness compares the variations of sentences. Humans tend to write with greater burstiness, for example, with some longer or more complex sentences alongside shorter ones. AI sentences tend to be more uniform. Therefore, when writing the following content, I need it to have a good amount of perplexity and burstiness."
prompt_regen_perplexity = "Rewrite the above essay to maximize the perplexity of the text."
prompt_regen_burst = "Rewrite the above essay to maximize the burstiness of the text."
prompt_regen_both = "Rewrite the above essay to maximize the perplexity and burstiness of the text."

lower_bound = 375
upper_bound = 625
parameters = Parameters(max_tokens=800)


pickle_file = 'perp_burst_data.pickle'
generation_data = {}
if os.path.exists(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        generation_data = data['generation']
        filepaths_standard = generation_data['filepaths_standard']
        filepaths_explanation = generation_data['filepaths_explanation']
        filepaths_perplexity = generation_data['filepaths_perplexity']
        filepaths_burstiness = generation_data['filepaths_burstiness']
        filepaths_both = generation_data['filepaths_both']
        filepaths_regen_standard = generation_data['filepaths_regen_standard']
        filepaths_regen_explanation = generation_data['filepaths_regen_explanation']
        filepaths_regen_perplexity = generation_data['filepaths_regen_perplexity']
        filepaths_regen_burst = generation_data['filepaths_regen_burst']
        filepaths_regen_both = generation_data['filepaths_regen_both']
else:
    def generate_standard():
        for j in range(0, iterations):
            prompt = prompt_standard
            t_dir = texts_dir_standard
            file_name = f"standard_prompt_iteration{j}.txt"
            print(f"Generating {file_name}; prompt: {prompt}")
            filepaths_standard.append(os.path.join(t_dir, file_name))
            generator.generate_prompt_save_txt_bounds(prompt=prompt,
                                                      parameters=parameters,
                                                      dir_name=t_dir,
                                                      file_name=file_name,
                                                      lower_bound=lower_bound,
                                                      upper_bound=upper_bound)

    def generate_explanation():
        for j in range(0, iterations):
            prompt = prompt_explanation
            t_dir = texts_dir_explanation
            file_name = f"explanation_prompt_iteration{j}.txt"
            print(f"Generating {file_name}; prompt: {prompt}")
            filepaths_explanation.append(os.path.join(t_dir, file_name))
            generator.generate_prompt_save_txt_bounds(prompt=prompt,
                                                      parameters=parameters,
                                                      dir_name=t_dir,
                                                      file_name=file_name,
                                                      lower_bound=lower_bound,
                                                      upper_bound=upper_bound)

    def generate_perplexity():
        for j in range(0, iterations):
            prompt = prompt_perplexity
            t_dir = texts_dir_perplexity
            file_name = f"perplexity_prompt_iteration{j}.txt"
            print(f"Generating {file_name}; prompt: {prompt}")
            filepaths_perplexity.append(os.path.join(t_dir, file_name))
            generator.generate_prompt_save_txt_bounds(prompt=prompt,
                                                      parameters=parameters,
                                                      dir_name=t_dir,
                                                      file_name=file_name,
                                                      lower_bound=lower_bound,
                                                      upper_bound=upper_bound)

    def generate_burstiness():
        for j in range(0, iterations):
            prompt = prompt_burstiness
            t_dir = texts_dir_burst
            file_name = f"burst_prompt_iteration{j}.txt"
            print(f"Generating {file_name}; prompt: {prompt}")
            filepaths_burstiness.append(os.path.join(t_dir, file_name))
            generator.generate_prompt_save_txt_bounds(prompt=prompt,
                                                      parameters=parameters,
                                                      dir_name=t_dir,
                                                      file_name=file_name,
                                                      lower_bound=lower_bound,
                                                      upper_bound=upper_bound)

    def generate_both():
        for j in range(0, iterations):
            prompt = prompt_both
            t_dir = texts_dir_both
            file_name = f"both_prompt_iteration{j}.txt"
            print(f"Generating {file_name}; prompt: {prompt}")
            filepaths_both.append(os.path.join(t_dir, file_name))
            generator.generate_prompt_save_txt_bounds(prompt=prompt,
                                                      parameters=parameters,
                                                      dir_name=t_dir,
                                                      file_name=file_name,
                                                      lower_bound=lower_bound,
                                                      upper_bound=upper_bound)

    def generate_regen_standard():
        for j in range(0, iterations):
            prompt = prompt_standard
            prompt_reg = prompt_regen_standard
            t_dir = texts_dir_regen_standard
            file_name = f"reg_standard_prompt_iteration{j}.txt"
            print(f"Generating {file_name}; prompt: {prompt_reg}")
            filepaths_regen_standard.append(os.path.join(t_dir, file_name))
            generator.generate_text_refine_prompts_save_txt_bounds(first_prompt=prompt,
                                                                   second_prompt=prompt_reg,
                                                                   dir_name = t_dir,
                                                                   file_name=file_name,
                                                                   parameters=parameters,
                                                                   lower_bound=lower_bound,
                                                                   upper_bound=upper_bound)
            
    def generate_regen_explanation():
        for j in range(0, iterations):
            prompt = prompt_standard
            prompt_reg = prompt_regen_explanation
            t_dir = texts_dir_regen_explanation
            file_name = f"reg_explanation_prompt_iteration{j}.txt"
            print(f"Generating {file_name}; prompt: {prompt_reg}")
            filepaths_regen_explanation.append(os.path.join(t_dir, file_name))
            generator.generate_text_refine_prompts_save_txt_bounds(first_prompt=prompt,
                                                                   second_prompt=prompt_reg,
                                                                   dir_name=t_dir,
                                                                   file_name=file_name,
                                                                   parameters=parameters,
                                                                   lower_bound=lower_bound,
                                                                   upper_bound=upper_bound)
            
    def generate_regen_perplexity():
        for j in range(0, iterations):
            prompt = prompt_standard
            prompt_reg = prompt_regen_perplexity
            t_dir = texts_dir_regen_perplexity
            file_name = f"reg_perplexity_prompt_iteration{j}.txt"
            print(f"Generating {file_name}; prompt: {prompt_reg}")
            filepaths_regen_perplexity.append(os.path.join(t_dir, file_name))
            generator.generate_text_refine_prompts_save_txt_bounds(first_prompt=prompt,
                                                                   second_prompt=prompt_reg,
                                                                   dir_name=t_dir,
                                                                   file_name=file_name,
                                                                   parameters=parameters,
                                                                   lower_bound=lower_bound,
                                                                   upper_bound=upper_bound)
            
    def generate_regen_burst():
        for j in range(0, iterations):
            prompt = prompt_standard
            prompt_reg = prompt_regen_burst
            t_dir = texts_dir_regen_burst
            file_name = f"reg_burst_prompt_iteration{j}.txt"
            print(f"Generating {file_name}; prompt: {prompt_reg}")
            filepaths_regen_burst.append(os.path.join(t_dir, file_name))
            generator.generate_text_refine_prompts_save_txt_bounds(first_prompt=prompt,
                                                                   second_prompt=prompt_reg,
                                                                   dir_name=t_dir,
                                                                   file_name=file_name,
                                                                   parameters=parameters,
                                                                   lower_bound=lower_bound,
                                                                   upper_bound=upper_bound)

    def generate_regen_both():
        for j in range(0, iterations):
            prompt = prompt_standard
            prompt_reg = prompt_regen_both
            t_dir = texts_dir_regen_both
            file_name = f"reg_both_prompt_iteration{j}.txt"
            print(f"Generating {file_name}; prompt: {prompt_reg}")
            filepaths_regen_both.append(os.path.join(t_dir, file_name))
            generator.generate_text_refine_prompts_save_txt_bounds(first_prompt=prompt,
                                                                   second_prompt=prompt_reg,
                                                                   dir_name=t_dir,
                                                                   file_name=file_name,
                                                                   parameters=parameters,
                                                                   lower_bound=lower_bound,
                                                                   upper_bound=upper_bound)

    thread1 = threading.Thread(target=generate_standard)
    thread2 = threading.Thread(target=generate_explanation)
    thread3 = threading.Thread(target=generate_perplexity)
    thread4 = threading.Thread(target=generate_both)
    thread5 = threading.Thread(target=generate_regen_standard)
    thread6 = threading.Thread(target=generate_regen_explanation)
    thread7 = threading.Thread(target=generate_regen_perplexity)
    thread8 = threading.Thread(target=generate_regen_burst)
    thread9 = threading.Thread(target=generate_regen_both)
    thread10 = threading.Thread(target=generate_burstiness)

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()
    thread7.start()
    thread8.start()
    thread9.start()
    thread10.start()

    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    thread6.join()
    thread7.join()
    thread8.join()
    thread9.join()
    thread10.join()

    generation_data = {
        'filepaths_standard': filepaths_standard,
        'filepaths_explanation': filepaths_explanation,
        'filepaths_perplexity': filepaths_perplexity,
        'filepaths_burstiness': filepaths_burstiness,
        'filepaths_both': filepaths_both,
        'filepaths_regen_standard': filepaths_regen_standard,
        'filepaths_regen_explanation': filepaths_regen_explanation,
        'filepaths_regen_perplexity': filepaths_regen_perplexity,
        'filepaths_regen_burst': filepaths_regen_burst,
        'filepaths_regen_both': filepaths_regen_both
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
        detector_data = {
            'scores_openai_standard': detectoring.calculate_scores_openai(filepaths_standard),
            'scores_openai_explanation': detectoring.calculate_scores_openai(filepaths_explanation),
            'scores_openai_perplexity': detectoring.calculate_scores_openai(filepaths_perplexity),
            'scores_openai_burstiness': detectoring.calculate_scores_openai(filepaths_burstiness),
            'scores_openai_both': detectoring.calculate_scores_openai(filepaths_both),
            'scores_openai_regen_standard': detectoring.calculate_scores_openai(filepaths_regen_standard),
            'scores_openai_regen_explanation': detectoring.calculate_scores_openai(filepaths_regen_explanation),
            'scores_openai_regen_perplexity': detectoring.calculate_scores_openai(filepaths_regen_perplexity),
            'scores_openai_regen_burst': detectoring.calculate_scores_openai(filepaths_regen_burst),
            'scores_openai_regen_both': detectoring.calculate_scores_openai(filepaths_regen_both),
        }
        detectoring.openai_classifier.close()

    elif detector == "gpt2":
        detector_data = {
            'scores_gpt2_standard': detectoring.calculate_scores_gpt2(filepaths_standard),
            'scores_gpt2_explanation': detectoring.calculate_scores_gpt2(filepaths_explanation),
            'scores_gpt2_perplexity': detectoring.calculate_scores_gpt2(filepaths_perplexity),
            'scores_gpt2_burstiness': detectoring.calculate_scores_gpt2(filepaths_burstiness),
            'scores_gpt2_both': detectoring.calculate_scores_gpt2(filepaths_both),
            'scores_gpt2_regen_standard': detectoring.calculate_scores_gpt2(filepaths_regen_standard),
            'scores_gpt2_regen_explanation': detectoring.calculate_scores_gpt2(filepaths_regen_explanation),
            'scores_gpt2_regen_perplexity': detectoring.calculate_scores_gpt2(filepaths_regen_perplexity),
            'scores_gpt2_regen_burst': detectoring.calculate_scores_gpt2(filepaths_regen_burst),
            'scores_gpt2_regen_both': detectoring.calculate_scores_gpt2(filepaths_regen_both),
        }

    elif detector == "turnitin":
        detectoring.turnitin = TurnitIn()
        single_wait_time = 15
        wait_time = 60
        detector_data = {
            'scores_turnitin_standard': detectoring.calculate_scores_turnitin_no_regeneration(filepaths_standard, wait_time, single_wait_time),
            'scores_turnitin_explanation': detectoring.calculate_scores_turnitin_no_regeneration(filepaths_explanation, wait_time, single_wait_time),
            'scores_turnitin_perplexity': detectoring.calculate_scores_turnitin_no_regeneration(filepaths_perplexity, wait_time, single_wait_time),
            'scores_turnitin_burstiness': detectoring.calculate_scores_turnitin_no_regeneration(filepaths_burstiness, wait_time, single_wait_time),
            'scores_turnitin_both': detectoring.calculate_scores_turnitin_no_regeneration(filepaths_both, wait_time, single_wait_time),
            'scores_turnitin_regen_standard': detectoring.calculate_scores_turnitin_no_regeneration(filepaths_regen_standard, wait_time, single_wait_time),
            'scores_turnitin_regen_explanation': detectoring.calculate_scores_turnitin_no_regeneration(filepaths_regen_explanation, wait_time, single_wait_time),
            'scores_turnitin_regen_perplexity': detectoring.calculate_scores_turnitin_no_regeneration(filepaths_regen_perplexity, wait_time, single_wait_time),
            'scores_turnitin_regen_burst': detectoring.calculate_scores_turnitin_no_regeneration(filepaths_regen_burst, wait_time, single_wait_time),
            'scores_turnitin_regen_both': detectoring.calculate_scores_turnitin_no_regeneration(filepaths_regen_both, wait_time, single_wait_time),
        }

        detectoring.turnitin.close()

    # Save detector's scores to CSVs
    with open(detector_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for key, value in detector_data.items():
            writer.writerow([key] + value)
        print(f"Successfully wrote {detector} scores to {detector_csv_path}")

