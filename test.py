import time

from detectors.turnitin_detector import TurnitIn

turnitin = TurnitIn()
turnitin.delete_all_uploaded_files()