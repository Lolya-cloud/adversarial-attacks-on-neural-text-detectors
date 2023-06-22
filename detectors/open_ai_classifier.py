from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from enum import Enum


class Classification(Enum):
    VERY_UNLIKELY = 1
    UNLIKELY = 2
    UNCLEAR = 3
    POSSIBLY = 4
    LIKELY = 5


class OpenAiClassifier:

    def __init__(self):
        self.classifier_path = "https://platform.openai.com/ai-text-classifier"
        self.textarea_class = "text-input text-input-md detector-input-textarea"
        self.submit_button_class = "btn btn-sm btn-filled btn-primary"
        self.output_class = "detector-output"
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        self.driver.get(self.classifier_path)
        WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, "." + ".".join(self.textarea_class.split()))))
        self.textarea = self.find_element_by_class_name(self.textarea_class)
        self.submit_button = self.find_element_by_class_name(self.submit_button_class)

        self.score_mapping = {
            "very unlikely": 0.05,
            "unlikely": 0.275,
            "unclear if it is": 0.675,
            "possibly": 0.94,
            "likely": 0.99
        }

    def find_element_by_class_name(self, class_name):
        class_names = class_name.split()
        elements = self.driver.find_elements(By.CSS_SELECTOR, "." + ".".join(class_names))
        if elements:
            return elements[0]
        else:
            return None

    def submit_and_scrape(self, text):
        while True:
            self.textarea.clear()
            self.textarea.send_keys(text)
            time.sleep(3)
            self.submit_button.click()
            time.sleep(4)
            output = self.find_element_by_class_name(self.output_class)
            result_text = output.text.lower()  # Convert to lowercase for comparison

            for keyword, score in self.score_mapping.items():
                if keyword in result_text:
                    return score

    def submit_and_scrape_label(self, text):
        while True:
            self.textarea.clear()
            self.textarea.send_keys(text)
            time.sleep(3)
            self.submit_button.click()
            time.sleep(4)
            output = self.find_element_by_class_name(self.output_class)
            result_text = output.text.lower()  # Convert to lowercase for comparison

            for keyword, score in self.score_mapping.items():
                if keyword in result_text:
                    return keyword

    def close(self):
        self.driver.quit()
