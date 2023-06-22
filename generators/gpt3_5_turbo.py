import openai
from utilities.TXTHandler import TXTHandler
import time
from openai.error import OpenAIError


class ChatGPT:
    def __init__(self):
        openai.api_key = "empty"
        self.modelId = 'gpt-3.5-turbo'

    def chat_gpt_conversation(self, conversation, parameters):
        response = openai.ChatCompletion.create(
            model=self.modelId,
            messages=conversation,
            temperature=parameters.temperature,
            top_p=parameters.top_p,
            n=parameters.n,
            max_tokens=parameters.max_tokens,
            presence_penalty=parameters.presence_penalty,
            frequency_penalty=parameters.frequency_penalty
        )
        conversation.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
        return conversation

    def generate_prompt(self, question_text, parameters):
        conversation = []
        prompt = question_text
        conversation.append({'role': 'user', 'content': prompt})
        while True:
            try:
                # Place your code here. For example:
                conversation = self.chat_gpt_conversation(conversation, parameters)
                break  # If the code execution reaches this line, no exception was raised and the loop will be exited
            except openai.error.RateLimitError:
                print("Rate limit error encountered. Sleeping and retrying...")
                time.sleep(60)
            except OpenAIError as e:
                print(f"An OpenAI error occurred: {e}. Retrying...")
                time.sleep(10)

        formatted_promt = '{0}: {1}\n'.format(conversation[-1]['role'].strip(), conversation[-1]['content'].strip())
        if formatted_promt.startswith("assistant:"):
            formatted_promt = formatted_promt[len("assistant:"):].strip()

        return formatted_promt

    def generate_prompt_save_txt(self, prompt, parameters, dir_name, file_name):
        text = self.generate_prompt(prompt, parameters)

        # Count the number of words in the generated text
        num_words = len(text.split())

        # Check if the number of words falls within the 450-550 words range
        while num_words < 450 or num_words > 550:
            print(f"Number of words insufficient: {num_words}. Regenerating")
            # Regenerate the text if the number of words is outside the desired range
            text = self.generate_prompt(prompt, parameters)
            num_words = len(text.split())

        txt_handler = TXTHandler(dir_name)
        txt_handler.save_text_to_txt(text, file_name)

    def generate_prompt_save_txt_bounds(self, prompt, parameters, dir_name, file_name, lower_bound, upper_bound):
        text = self.generate_prompt(prompt, parameters)

        # Count the number of words in the generated text
        num_words = len(text.split())

        # Check if the number of words falls within the 450-550 words range
        while num_words <= lower_bound or num_words >= upper_bound:
            print(f"Number of words insufficient: {num_words}. Regenerating")
            # Regenerate the text if the number of words is outside the desired range
            text = self.generate_prompt(prompt, parameters)
            num_words = len(text.split())

        txt_handler = TXTHandler(dir_name)
        filepath = txt_handler.save_text_to_txt(text, file_name)
        return filepath

    def generate_text_refine_prompts_save_txt_bounds(self, first_prompt, second_prompt, dir_name, file_name, parameters, lower_bound, upper_bound):
        text = self.generate_text_refine_prompts(first_prompt, second_prompt, parameters)

        # Count the number of words in the generated text
        num_words = len(text.split())

        # Check if the number of words falls within the 450-550 words range
        while num_words <= lower_bound or num_words >= upper_bound:
            print(f"Number of words insufficient: {num_words}. Regenerating")
            # Regenerate the text if the number of words is outside the desired range
            text = self.generate_text_refine_prompts(first_prompt, second_prompt, parameters)
            num_words = len(text.split())

        txt_handler = TXTHandler(dir_name)
        filepath = txt_handler.save_text_to_txt(text, file_name)
        return filepath

    def generate_text_refine_prompts(self, first_prompt, second_prompt, parameters):
        conversation = []
        while True:
            try:
                # First prompt
                conversation.append({'role': 'user', 'content': first_prompt})
                conversation = self.chat_gpt_conversation(conversation, parameters)
                break  # If the code execution reaches this line, no exception was raised and the loop will be exited
            except openai.error.RateLimitError:
                print("Rate limit error encountered. Sleeping and retrying...")
                time.sleep(60)
            except OpenAIError as e:
                print(f"An OpenAI error occurred: {e}. Retrying...")
                time.sleep(10)

        while True:
            try:
                # Second prompt
                conversation.append({'role': 'user', 'content': second_prompt})
                conversation = self.chat_gpt_conversation(conversation, parameters)
                break  # If the code execution reaches this line, no exception was raised and the loop will be exited
            except openai.error.RateLimitError:
                print("Rate limit error encountered. Sleeping and retrying...")
                time.sleep(60)
            except OpenAIError as e:
                print(f"An OpenAI error occurred: {e}. Retrying...")
                time.sleep(10)
        formatted_prompt = '{0}: {1}\n'.format(conversation[-1]['role'].strip(), conversation[-1]['content'].strip())
        if formatted_prompt.startswith("assistant:"):
            formatted_prompt = formatted_prompt[len("assistant:"):].strip()
        return formatted_prompt

