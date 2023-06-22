import os


class TXTHandler:
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def save_text_to_txt(self, text, file_name):
        file_path = os.path.join(self.dir_path, f'{file_name}')
        text = text.replace('\t', '')
        text = text.replace("\\", "")
        text = text.lstrip('\n')
        # Open the file in write mode with UTF-8 encoding
        with open(file_path, "w", encoding="utf-8") as file:
            # Write the text
            file.write(text)

        return file_path


class TXTLoader:

    @staticmethod
    def load_file(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:  # Open each file
            file_content = file.read()  # Read the content of the file
        return file_content
