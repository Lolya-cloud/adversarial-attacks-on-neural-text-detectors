import openai
from utilities.TXTHandler import TXTHandler


# noinspection SpellCheckingInspection
class TextDavinci:
    def __init__(self):
        self.api_key = "empty"
        self.modelId = 'text-davinci-003'
        openai.api_key = self.api_key

    def generate_prompt(self, prompt, parameters):
        completions = openai.Completion.create(
            engine=self.modelId,
            prompt=prompt,
            **parameters.to_dict()
        )
        return completions.choices[0].text

    def generate_prompt_save_txt(self, prompt, parameters, dir_name, file_name):
        text = self.generate_prompt(prompt, parameters)
        txt_handler = TXTHandler(dir_name)
        txt_handler.save_text_to_txt(text, file_name)

