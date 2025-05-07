# Import necessary modules
import os  # For handling file paths
import joblib  # For loading model and normalizer files
from file_structure import file_structure  # Importing the predefined file structure

# Function to load all models, normalizers, and labels from different categories
def load_all_categories(base_path="final_models"):
    all_loaded = {}  # Dictionary to store all loaded data

    # Loop through each category in the file structure
    for category, files in file_structure.items():
        loaded_data = {"models": [], "normalizers": [], "labels": []}  # Initialize storage for this category
        
        # Load model files using joblib
        for model_file in files["models"]:
            model_path = os.path.join(base_path, category, 'models', model_file)  # Build the full path to the model file
            loaded_data["models"].append(joblib.load(model_path))  # Load and append the model

        # Load normalizer files using joblib
        for norm_file in files["normalizers"]:
            norm_path = os.path.join(base_path, category, 'normalizers', norm_file)  # Build the full path to the normalizer file
            loaded_data["normalizers"].append(joblib.load(norm_path))  # Load and append the normalizer

        # Load label files by reading text content
        for label_file in files["labels"]:
            label_path = os.path.join(base_path, category, 'labels', label_file)  # Build the full path to the label file
            with open(label_path, "r") as f:

                # loaded_data["labels"].append([line.strip() for line in f if line.strip()])

                # Read all lines from the label file, removing newline characters
                loaded_data["labels"].append(f.read().splitlines())

        # Save loaded data under the corresponding category
        all_loaded[category] = loaded_data

    return all_loaded  # Return the complete dictionary with all loaded items
