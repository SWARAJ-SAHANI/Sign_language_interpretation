# Define a dictionary that holds file structures for different sign categories
file_structure = {
    "numbers": {  # Category: numbers
        "models": ["1_number_model.joblib", "2_number_model.joblib"],  # List of model files for numbers
        "normalizers": ["1_number_normalizer.joblib", "2_number_normalizer.joblib"],  # List of normalizer files for numbers
        "labels": ["1_number_label.txt", "2_number_label.txt"]  # List of label files for numbers
    },
    "alphabets": {  # Category: alphabets
        "models": ["1_alphabet_model.joblib", "2_alphabet_model.joblib"],  # List of model files for alphabets
        "normalizers": ["1_alphabet_normalizer.joblib", "2_alphabet_normalizer.joblib"],  # List of normalizer files for alphabets
        "labels": ["1_alphabet_label.txt", "2_alphabet_label.txt"]  # List of label files for alphabets
    },
    "simple_signs": {  # Category: simple signs
        "models": ["1_simple_model.joblib", "2_simple_model.joblib"],  # List of model files for simple signs
        "normalizers": ["1_simple_normalizer.joblib", "2_simple_normalizer.joblib"],  # List of normalizer files for simple signs
        "labels": ["1_simple_label.txt", "2_simple_label.txt"]  # List of label files for simple signs
    },
    "custom_signs": {  # Category: custom signs
        "models": ["1_custom_model.joblib", "2_custom_model.joblib"],  # List of model files for custom signs
        "normalizers": ["1_custom_normalizer.joblib", "2_custom_normalizer.joblib"],  # List of normalizer files for custom signs
        "labels": ["1_custom_label.txt", "2_custom_label.txt"]  # List of label files for custom signs
    }
}
