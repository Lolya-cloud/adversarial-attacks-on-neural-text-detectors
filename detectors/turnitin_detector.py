import selenium.common.exceptions
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from utilities.TXTHandler import TXTHandler
import time
import os


class TurnitIn():
    """
    A class for interacting with Turnitin.
    """

    def __init__(self):
        """
        The constructor for TurnitIn class.
        """
        self.email = "empty"
        self.password = "empty"
        self.turnitin_link = "https://utwente.turnitin.com/"
        self.dir_path = 'C:/Users/vital/PycharmProjects/M12Project/texts'
        self.file_num = 0
        # Initialize the PDFHandler class
        self.file_handler = TXTHandler(self.dir_path)
        # Initialize webdriver
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        self.driver.get(self.turnitin_link)
        # Log into the site and navigate to the needed page
        self.launch()
        # wait for login
        time.sleep(5)

    def login(self):
        """
        The function to log into Turnitin.
        """
        # Wait for the username and password inputs to be present, then enter credentials and submit the form
        WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.ID, "username"))).send_keys(
            self.email)
        WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.ID, "password"))).send_keys(
            self.password)
        WebDriverWait(self.driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//button[@type='submit']"))).click()


    def launch_detector(self):
        """
        The function to launch Turnitin Similarity.
        """
        # Click the button to launch Turnitin Similarity
        WebDriverWait(self.driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//button[text()=' Launch ']"))).click()

        # Get current window handle
        current_window = self.driver.current_window_handle

        # Close the current window and switch to the new window
        self.driver.close()
        self.driver.switch_to.window([window for window in self.driver.window_handles if window != current_window][0])

    def refresh(self):
        self.driver.refresh()

    def open_folder(self):
        """
        The function to open the needed folder in Turnitin.
        """
        # Click the folder
        WebDriverWait(self.driver, 30).until(
            EC.presence_of_element_located(
                (By.XPATH, "//a[@href='/originality/new/inbox/a253bd35-1352-47e3-af15-1d2334feffca']"))).click()

    def upload(self, file_paths, upload_wait_time):
        """
        The function to upload files to Turnitin.
        """
        # Click the upload button
        time.sleep(3)
        upload_button = WebDriverWait(self.driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//button[@data-test-id='upload-button']")))
        self.driver.execute_script("arguments[0].click();", upload_button)
        time.sleep(3)  # Wait for the upload input to appear
        print(file_paths)
        # Enter the file paths into the upload input
        WebDriverWait(self.driver, 30).until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, 'input[type="file"].ngx-file-drop__file-input'))).send_keys(
            '\n'.join(file_paths))

        time.sleep(3)  # Wait for the files to be processed

        # Submit the upload form
        WebDriverWait(self.driver, 30).until(EC.presence_of_element_located(
            (By.XPATH, "//button[@type='submit' and contains(@class, 'sc-ui-button-s')]"))).click()

        self.wait_for_upload(wait_time=upload_wait_time)

    def wait_for_upload(self, wait_time):
        """
        The function to pause execution for a given amount of time.
        """
        time.sleep(wait_time)  # Wait for the upload to finish
        self.driver.refresh()

    def scrape_result(self, file_paths):
        """
        The function to scrape results from Turnitin.
        """
        file_names = self.get_filenames(file_paths)
        original_window = self.driver.current_window_handle
        results = []
        for name in file_names:
            # Click the table element for the file
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.XPATH, f'//*[@title="{name}"]'))).click()

            # Switch to the new window
            WebDriverWait(self.driver, 10).until(EC.number_of_windows_to_be(2))
            new_window = [window for window in self.driver.window_handles if window != original_window][0]
            self.driver.switch_to.window(new_window)
            time.sleep(5)  # Wait for the results
            # Scrape the similarity score
            xpath = '//tii-aiw-button[@type="cv"]'
            element = WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.XPATH, xpath)))
            percentage = element.get_attribute("percent")
            results.append(int(percentage))

            # Return to the original window
            self.driver.switch_to.window(original_window)

            # Close the new window
            self.driver.switch_to.window(new_window)
            self.driver.close()

            # Ensure the driver is pointed at the original window again
            self.driver.switch_to.window(original_window)

        return results

    def scrape_results_list(self, filepaths):
        original_window = self.driver.current_window_handle
        scores = []
        self.switch_to_250_rows()
        print(filepaths)
        file_names = self.get_filenames(filepaths)
        print(file_names)
        for name in file_names:
            print(name)
            # Click the table element for the file
            for _ in range(3):  # Try up to three times
                try:
                    # Wait for the element to become clickable
                    element = WebDriverWait(self.driver, 30).until(
                        EC.element_to_be_clickable((By.XPATH, f'//span[contains(text(),"{name}")]')))

                    # Scroll the element into view
                    self.driver.execute_script("arguments[0].scrollIntoView();", element)

                    # Use JavaScript to perform the click
                    self.driver.execute_script("arguments[0].click();", element)

                    break
                except ElementClickInterceptedException:
                    # If the click fails, wait for two seconds before retrying
                    time.sleep(2)

            try:
                # Switch to the new window
                WebDriverWait(self.driver, 10).until(EC.number_of_windows_to_be(2))
                new_window = [window for window in self.driver.window_handles if window != original_window][0]
                self.driver.switch_to.window(new_window)
                time.sleep(5)  # Wait for the results
            except selenium.common.exceptions.TimeoutException:
                print("Document not processed, sleeping for 3 minutes")
                time.sleep(180)
                self.refresh()
                for _ in range(3):  # Try up to three times
                    try:
                        # Wait for the element to become clickable
                        element = WebDriverWait(self.driver, 30).until(
                            EC.element_to_be_clickable((By.XPATH, f'//span[contains(text(),"{name}")]')))

                        # Scroll the element into view
                        self.driver.execute_script("arguments[0].scrollIntoView();", element)

                        # Use JavaScript to perform the click
                        self.driver.execute_script("arguments[0].click();", element)

                        break
                    except ElementClickInterceptedException:
                        # If the click fails, wait for two seconds before retrying
                        time.sleep(2)
                try:
                    # Switch to the new window
                    WebDriverWait(self.driver, 10).until(EC.number_of_windows_to_be(2))
                    new_window = [window for window in self.driver.window_handles if window != original_window][0]
                    self.driver.switch_to.window(new_window)
                    time.sleep(5)  # Wait for the results
                except selenium.common.exceptions.TimeoutException:
                    print("still not processed, returning none and continuing to the next iteration")
                    scores.append(None)
                    continue

            # Scrape the similarity score
            xpath = '//tii-aiw-button[@type="cv"]'
            element = WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.XPATH, xpath)))
            percentage = element.get_attribute("percent")
            if percentage is not None:
                scores.append(int(percentage))
            else:
                scores.append(None)

            # Return to the original window
            self.driver.switch_to.window(original_window)

            # Close the new window
            self.driver.switch_to.window(new_window)
            self.driver.close()

            # Ensure the driver is pointed at the original window again
            self.driver.switch_to.window(original_window)
            self.refresh()
        print(scores)
        self.delete_all_uploaded_files()
        return scores

    def scrape_results_dictionary(self, file_paths_dict):
        """
           The function to scrape results from Turnitin.
        """
        original_window = self.driver.current_window_handle
        results_dict = {}
        self.switch_to_250_rows()
        print(file_paths_dict)
        for param_value, file_paths in file_paths_dict.items():
            file_names = self.get_filenames(file_paths)
            param_results = []
            for name in file_names:
                print(name)
                # Click the table element for the file
                for _ in range(3):  # Try up to three times
                    try:
                        # Wait for the element to become clickable
                        element = WebDriverWait(self.driver, 30).until(
                            EC.element_to_be_clickable((By.XPATH, f'//span[contains(text(),"{name}")]')))

                        # Scroll the element into view
                        self.driver.execute_script("arguments[0].scrollIntoView();", element)

                        # Use JavaScript to perform the click
                        self.driver.execute_script("arguments[0].click();", element)

                        break
                    except ElementClickInterceptedException:
                        # If the click fails, wait for two seconds before retrying
                        time.sleep(2)

                # Switch to the new window
                WebDriverWait(self.driver, 10).until(EC.number_of_windows_to_be(2))
                new_window = [window for window in self.driver.window_handles if window != original_window][0]
                self.driver.switch_to.window(new_window)
                time.sleep(5)  # Wait for the results

                # Scrape the similarity score
                xpath = '//tii-aiw-button[@type="cv"]'
                element = WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.XPATH, xpath)))
                percentage = element.get_attribute("percent")
                if percentage is not None:
                    param_results.append(int(percentage))
                else:
                    param_results.append(None)

                # Return to the original window
                self.driver.switch_to.window(original_window)

                # Close the new window
                self.driver.switch_to.window(new_window)
                self.driver.close()

                # Ensure the driver is pointed at the original window again
                self.driver.switch_to.window(original_window)
                self.refresh()

            results_dict[param_value] = param_results

        print(results_dict)
        self.delete_all_uploaded_files()
        return results_dict

    def delete_all_uploaded_files(self):
        try:
            # Wait for the page to load
            WebDriverWait(self.driver, 30).until(EC.visibility_of_element_located((By.TAG_NAME, 'body')))

            # Select the checkbox
            checkbox = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located(
                    (By.ID, 'wi-checkbox-1-input')))
            if checkbox.is_enabled() and checkbox.is_displayed():
                self.driver.execute_script("arguments[0].click();", checkbox)
            else:
                print("Checkbox is not clickable or not displayed")
            time.sleep(3)
            # Click the delete button
            delete_button = WebDriverWait(self.driver, 30).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[data-test-id="trash-selected-button"]')))
            if delete_button.is_enabled() and delete_button.is_displayed():
                delete_button.click()
            else:
                print("Delete button is not clickable or not displayed")
            # wait for the deletion
            time.sleep(3)

        except TimeoutException:
            print("Timeout Exception: Element not found")
        except NoSuchElementException:
            print("No such element exception: Element not found")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def switch_to_250_rows(self):
        """
        This function switches the page view to display 250 rows.
        """
        # Attempt to find the button whose span is "250"
        try:
            dropdown_button = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '//button[@class="mat-menu-trigger"]/span[text()="250"]')))
        except TimeoutException:
            dropdown_button = None

        # If the button's span is not "250", switch it to "250"
        if dropdown_button is None:
            # Find the button whose span is "25"
            dropdown_button = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '//button[@class="mat-menu-trigger"]/span[text()="25"]')))

            # Click the dropdown menu button
            dropdown_button.click()

            # Wait for the dropdown menu to appear and select "250" from the list
            WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located(
                    (By.XPATH, '//button[@mat-menu-item and contains(text(), "250")]'))).click()

    def submit_and_scrape(self, texts, wait_time):
        """
        The function to upload files and scrape the results.
        """
        file_paths = self.save_to_txt(texts)
        self.upload(file_paths, wait_time)
        return self.scrape_result(file_paths)

    def submit_and_scrape_existing_files_list(self, file_paths_list, wait_time):
        self.upload(file_paths_list, wait_time)
        return self.scrape_results_list(file_paths_list)

    def submit_and_scrape_existing_files(self, file_paths_dict, wait_time):
        all_file_paths = []
        for file_paths in file_paths_dict.values():
            all_file_paths.extend(file_paths)
        print(all_file_paths)
        print(file_paths_dict)
        self.upload(all_file_paths, wait_time)
        return self.scrape_results_dictionary(file_paths_dict)

    def save_to_txt(self, texts):
        # Create the PDF files and get their file paths
        file_paths = []
        for i, text in enumerate(texts, start=1):
            self.file_num += 1
            file_path = self.file_handler.save_text_to_txt(text, f'{self.file_num}.txt')
            file_paths.append(file_path)
        return file_paths

    def launch(self):
        """
        The function to log into the site and navigate to the needed page.
        """
        self.login()
        self.launch_detector()
        self.open_folder()

    def get_filenames(self, file_paths):
        """
        The function to get filenames from file paths.
        """
        return [os.path.basename(path) for path in file_paths]

    def close(self):
        """
        The function to quit the webdriver.
        """
        self.driver.close()
