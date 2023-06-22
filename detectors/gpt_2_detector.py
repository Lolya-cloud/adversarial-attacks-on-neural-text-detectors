import requests
from urllib.parse import quote
import json


class GPT2Output:
    def __init__(self, all_tokens, used_tokens, real_probability, fake_probability):
        self.all_tokens = all_tokens
        self.used_tokens = used_tokens
        self.real_probability = real_probability
        self.fake_probability = fake_probability


class GPT2Detector:
    def __init__(self):
        self.url = 'https://openai-openai-detector--g5s42.hf.space/'

    def submit_and_scrape(self, text):
        quoted_text = quote(text)
        full_url = self.url + '?' + quoted_text
        try:
            response = requests.get(full_url)
            response.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            raise Exception("Http Error:", errh)
        except requests.exceptions.ConnectionError as errc:
            raise Exception("Error Connecting:", errc)
        except requests.exceptions.Timeout as errt:
            raise Exception("Timeout Error:", errt)
        except requests.exceptions.RequestException as err:
            raise Exception("Something went wrong: ", err)

        try:
            data = json.loads(response.text)
            return GPT2Output(data['all_tokens'], data['used_tokens'],
                              data['real_probability'], data['fake_probability'])
        except json.JSONDecodeError as e:
            raise Exception('Failed to decode JSON: ', e)
        except KeyError as e:
            raise Exception('JSON did not have expected format: ', e)