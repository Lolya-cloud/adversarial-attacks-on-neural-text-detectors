from bayes_opt import BayesianOptimization
from detectors.turnitin_detector import TurnitIn
from generators.text_davinchi import TextDavinci
from generators.openAiParameters import Parameters
import matplotlib.pyplot as plt
import json
import pandas as pd

def load_prompts(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    return content.split('\n\n')

train_prompts = load_prompts('../prompts/train_prompts_opt.txt')
test_prompts = load_prompts('../prompts/test_prompts_opt.txt')

turnitin = TurnitIn()
generator = TextDavinci()

# Initialize a pandas DataFrame for storing evaluation data
eval_data = pd.DataFrame(columns=['Prompt Index', 'Temperature', 'Top_p', 'Presence Penalty', 'Frequency Penalty', 'Detector Score'])

def evaluate_parameters(temperature, top_p, presence_penalty, frequency_penalty):
    global eval_data  # Use the dataframe we declared above
    parameters = Parameters(max_tokens=500,
                            temperature=temperature,
                            top_p=top_p,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty)
    texts = [generator.generate_prompt(prompt, parameters) for prompt in train_prompts]
    print(f"Parameters: temperature={temperature}, top_p={top_p}, presence_penalty={presence_penalty}, frequency_penalty={frequency_penalty}")
    scores = turnitin.submit_and_scrape(texts, 20)
    print(scores)
    avg_score = -sum(scores) / len(scores)

    # Add results to the dataframe for each prompt
    for i in range(len(train_prompts)):
        eval_data = eval_data.append({
            'Prompt Index': i,
            'Temperature': temperature,
            'Top_p': top_p,
            'Presence Penalty': presence_penalty,
            'Frequency Penalty': frequency_penalty,
            'Detector Score': scores[i] if i < len(scores) else None
        }, ignore_index=True)

    # Save the evaluation data after each evaluation for backup
    eval_data.to_csv('evaluation_data.csv', index=False)

    return avg_score

pbounds = {'temperature': (0, 2), 'top_p': (0, 1), 'presence_penalty': (0, 2),
           'frequency_penalty': (0, 2)}

optimizer = BayesianOptimization(f=evaluate_parameters, pbounds=pbounds, random_state=1)
optimizer.maximize(init_points=2, n_iter=15)

print(optimizer.max)
with open('optimization_results.txt', 'w') as f:
    f.write(json.dumps(optimizer.max))

# Save the final dataframe at the end of optimization
eval_data.to_csv('final_evaluation_data.csv', index=False)

# Plot the optimization process
x = list(range(1, len(optimizer.res) + 1))
y = [res['target'] for res in optimizer.res]
plt.figure(figsize=(10, 5))
plt.plot(x, y, '-o')
plt.xlabel('Iteration')
plt.ylabel('Score')
plt.title('Convergence over time')
plt.show()

# Load the best parameters from the optimization
with open('optimization_results.txt', 'r') as f:
    best_parameters = json.loads(f.read())

# Convert values to their correct type
best_parameters['params']['max_tokens'] = int(best_parameters['params']['max_tokens'])
best_parameters['params']['n'] = int(best_parameters['params']['n'])
best_parameters['params']['logprobs'] = int(best_parameters['params']['logprobs']) if best_parameters['params']['logprobs'] > 0 else None
best_parameters['params']['best_of'] = int(best_parameters['params']['best_of'])

# Create an empty DataFrame to store the test results
test_results = pd.DataFrame(columns=['Prompt', 'Optimized Score', 'Default Score'])

# Test the optimized parameters
for prompt in test_prompts:
    # Use the optimized parameters
    parameters = Parameters(**best_parameters['params'])
    texts_opt = [generator.generate_prompt(prompt, parameters) for _ in range(5)]
    scores_opt = turnitin.submit_and_scrape(texts_opt, 30)

    # Use the default parameters
    parameters = Parameters()
    texts_def = [generator.generate_prompt(prompt, parameters) for _ in range(5)]
    scores_def = turnitin.submit_and_scrape(texts_def, 30)

    # Add the result to the DataFrame
    test_results = test_results.append({
        'Prompt': prompt,
        'Optimized Score': sum(scores_opt) / len(scores_opt),
        'Default Score': sum(scores_def) / len(scores_def),
    }, ignore_index=True)

# Save the test results to a CSV file
test_results.to_csv('test_results.csv', index=False)

# Print the test results
print(test_results)