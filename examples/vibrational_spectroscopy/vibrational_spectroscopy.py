import os
import hydragnn

filepath = os.path.join(os.path.dirname(__file__), "vibrational_spectroscopy.json")
hydragnn.run_training(filepath)
