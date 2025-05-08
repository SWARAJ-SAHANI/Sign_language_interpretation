import shutil                                                   # For high-level file operations (copying, moving, deleting files)
import os                                                       # For interacting with the operating system (like file paths)
import pandas as pd                                             # For data manipulation and analysis
from joblib import dump                                         # To save models or other objects to disk
from sklearn.preprocessing import MinMaxScaler, LabelEncoder    # For feature scaling and label encoding
from sklearn.metrics import accuracy_score                      # To measure model prediction accuracy
from sklearn.model_selection import train_test_split            # To split data into training and testing sets
from sklearn.svm import SVC                                     # Support Vector Classifier for machine learning tasks
from sklearn.ensemble import RandomForestClassifier             # Random Forest model for classification
from sklearn.linear_model import LogisticRegression             # Logistic Regression model for classification

# Set to True to load the CSV file only once and avoid reloading multiple times
load_once = True

# Initialize machine learning models with fixed random states for reproducibility
svm = SVC(random_state=42)                         # Support Vector Classifier
rf = RandomForestClassifier(random_state=42)       # Random Forest Classifier
lr = LogisticRegression(max_iter=1000, random_state=42)  # Logistic Regression with increased iterations

# Store the models in a dictionary for easy access and looping
models = {
    'SVC': svm,
    'LogisticRegression': lr,
    'RandomForest': rf
}

# Define source and destination directories for file movement
move_source = 'keypoints'
move_destination = 'data'

# Initialize dataframes and other storage variables
df1 = df2 = None
dataframes = []
labels = None

# Define the function to load custom keypoint data from CSV files
def load_data():
    global df1, df2, dataframes  # Declare global variables for dataframes
    df1 = df2 = None  # Initialize df1 and df2 as None to handle unassigned files
    dataframes = []  # Initialize an empty list to hold dataframes

    # Define file paths for two specific custom keypoint CSV files
    custom_path1 = 'keypoints/1_custom_keypoint.csv'
    custom_path2 = 'keypoints/2_custom_keypoint.csv'

    # Load CSV file from custom_path1 if the file exists
    if os.path.exists(custom_path1):
        df1 = pd.read_csv(custom_path1, on_bad_lines='skip')  # Read the CSV file into df1

    # Load CSV file from custom_path2 if the file exists
    if os.path.exists(custom_path2):
        df2 = pd.read_csv(custom_path2, on_bad_lines='skip')  # Read the CSV file into df2

    # Combine non-None dataframes into a list and assign it to the dataframes variable
    dataframes = [df for df in [df1, df2] if df is not None]

def clean_data():
    # Declare global variables so we can modify them inside this function
    global df1, df2, dataframes

    # Loop through each dataframe in the list
    for df in dataframes:
        # Remove rows with any missing values (NaNs)
        df.dropna(inplace=True)

        # Remove duplicate rows
        df.drop_duplicates(inplace=True)

        # Define a condition to identify rows with non-zero wrist coordinates
        # If the dataframe contains data for both wrists (left and right)
        if '1_WRIST_x' in df.columns:
            condition = (
                (df['0_WRIST_x'] != 0.0) | 
                (df['0_WRIST_y'] != 0.0) | 
                (df['1_WRIST_x'] != 0.0) | 
                (df['1_WRIST_y'] != 0.0)
            )
        else:
            # If only one wrist is present, just check that one
            condition = (
                (df['0_WRIST_x'] != 0.0) | 
                (df['0_WRIST_y'] != 0.0)
            )

        # Drop rows where any of the wrist coordinates are non-zero
        df.drop(df[condition].index, inplace=True)

        # Reset index after dropping rows to keep it clean and continuous
        df.reset_index(drop=True, inplace=True)

def create_label():
    # Declare global variables to access and modify them within the function
    global df1, df2, dataframes

    # Loop through each dataframe
    for df in dataframes:
        # Initialize label encoder
        le = LabelEncoder()

        # Encode string labels into integers and assign back to the 'label' column
        df['label'] = le.fit_transform(df['label'])

        # Determine the path to save class labels based on number of features (single or double hand)
        if len(df.columns) > 44:
            path = "final_models/custom_signs/labels/2_custom_label.txt"  # For two-hand keypoints
        else:
            path = "final_models/custom_signs/labels/1_custom_label.txt"  # For single-hand keypoints

        # if len(df.columns) > 44:
        #     path = "2_custom_label.txt"
        # else:
        #     path = "1_custom_label.txt" 

        # Write the class labels (original string form) to a text file
        with open(path, 'w') as f:
            for label in le.classes_:
                f.write(f"{label}\n")

def display_signs():
    # Access the global dataframes list
    global dataframes

    # Initialize an empty list to store unique labels
    labels_list = []

    # Loop through each dataframe
    for df in dataframes:
        # Loop through unique labels in the 'label' column
        for label in df['label'].unique():
            # Add label to the list if it's not already included
            if label not in labels_list:
                labels_list.append(label)

    # Return the list of unique labels found across all dataframes
    return labels_list

def remove_data(drop_label_index):
    # Access global variables
    global dataframes, labels

    # Get the label value to drop based on the provided index
    drop_label = labels[drop_label_index]

    # Loop through each dataframe and remove rows with the specified label
    for i in range(len(dataframes)):
        dataframes[i] = dataframes[i][dataframes[i]['label'] != drop_label]

# Define a function to move and rename files from source to destination
def move_file():
    # Define the path where files will be moved to
    path = 'data'

    # Counters to track how many single-hand and double-hand files already exist
    total_single_file = 0
    total_double_file = 0

    # First, count existing files in the destination directory to avoid overwriting
    for filename in os.listdir(path):
        if filename.startswith('1_custom_keypoint'):  # Check if the file is for single-hand data
            total_single_file += 1  # Increment counter for single-hand files
        if filename.startswith('2_custom_keypoint'):  # Check if the file is for double-hand data
            total_double_file += 1  # Increment counter for double-hand files

    # Move files from the source directory to the destination directory, renaming to avoid conflicts
    for filename in os.listdir(move_source):
        if filename.startswith('1_custom_keypoint'):  # Check if the file is for single-hand data
            # Set up source and new destination path with updated filename to avoid conflict
            source_path = os.path.join(move_source, filename)
            destination_path = os.path.join(move_destination, f"1_custom_data_{total_single_file+1}.csv")
            destination_original_path = os.path.join(move_destination, 'original', f"1_custom_data_original_{total_single_file+1}.csv")
            
            # Move the file to the original folder and save the DataFrame to a new file
            shutil.move(source_path, destination_original_path)
            df1.to_csv(destination_path)  # Save the dataframe to a new location

            total_single_file += 1  # Increment the counter for single-hand files

        elif filename.startswith('2_custom_keypoint'):  # Check if the file is for double-hand data
            # Set up source and new destination path with updated filename to avoid conflict
            source_path = os.path.join(move_source, filename)
            destination_path = os.path.join(move_destination, f"2_custom_data_{total_double_file+1}.csv")
            destination_original_path = os.path.join(move_destination, 'original', f"2_custom_data_original_{total_double_file+1}.csv")
            
            # Move the file to the original folder and save the DataFrame to a new file
            shutil.move(source_path, destination_original_path)
            df1.to_csv(destination_path)  # Save the dataframe to a new location

            total_double_file += 1  # Increment the counter for double-hand files

def evaluate_model(x_train_norm, x_test_norm, y_train, y_test):
    """
    Trains and evaluates multiple models on the given dataset,
    and returns the model with the highest accuracy.
    """

    best_model = None           # To store the best performing model
    best_accuracy = 0.0         # To track the highest accuracy achieved

    # Loop through all models defined in the global 'models' dictionary
    for model in models.values():
        # Train the model on the training data
        model.fit(x_train_norm, y_train)

        # Predict labels for the test data
        y_pred = model.predict(x_test_norm)

        # Calculate accuracy for the model
        accuracy = accuracy_score(y_test, y_pred)

        # Update best model if current model has higher accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    # Return the best performing model
    return best_model

def train_model():
    # Access global dataframes
    global df1, df2, dataframes

    # Loop through each dataset (assumed to be for different hand types)
    for df in dataframes:
        # Split features and labels
        x = df.drop(columns=['label'])
        y = df['label']
        
        # Split data into training and testing sets (15% for testing)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
        
        # Normalize the feature values using MinMaxScaler
        minmax = MinMaxScaler()
        x_train_norm = minmax.fit_transform(x_train)
        x_test_norm = minmax.transform(x_test)
        
        # Train and evaluate multiple models to select the best one
        best_model = evaluate_model(x_train_norm, x_test_norm, y_train, y_test)

        # Check if the dataframe has double-hand keypoints (more than 44 features)
        if len(df.columns) > 44:
            # Save the best model and normalizer for double-hand gestures
            dump(best_model, "final_models/custom_signs/models/2_custom_model.joblib")
            dump(minmax, "final_models/custom_signs/normalizers/2_custom_normalizer.joblib")
        else:
            # Save the best model and normalizer for single-hand gestures
            dump(best_model, "final_models/custom_signs/models/1_custom_model.joblib")
            dump(minmax, "final_models/custom_signs/normalizers/1_custom_normalizer.joblib")

        # if len(df.columns) > 44:
        #     # Save the best model and normalizer for double-hand gestures
        #     dump(best_model, "2_custom_model.joblib")
        #     dump(minmax, "2_custom_normalizer.joblib")
        # else:
        #     # Save the best model and normalizer for single-hand gestures
        #     dump(best_model, "1_custom_model.joblib")
        #     dump(minmax, "1_custom_normalizer.joblib")

# Define the main function with optional parameters for button click and removal index
def main(train_button=False, remove_index=None):
    global df1, df2, dataframes, labels, load_once  # Declare global variables to be used inside the function

    # Load data if it hasn't been loaded already or if there are fewer than 2 labels
    if load_once or (len(labels) < 2):
        load_data()  # Load the data (e.g., CSV files)
        load_once = False  # Mark that the data has been loaded

    labels = display_signs()  # Display available labels (for signs)

    # Check if there are no dataframes or labels are empty or None
    if (not dataframes) or (len(labels) == 0) or (labels is None):
        return 'no Data'  # Return a message indicating no data is available

    # If a removal index is provided, remove the data at that index
    if remove_index is not None:
        remove_data(remove_index)  # Replace with your actual removal logic to delete data
        labels = display_signs()  # Refresh the labels after removal

        # If no labels are left after removal, return a message indicating no data
        if len(labels) == 0:
            return 'no Data'
        return f"Data at index {remove_index} removed"  # Return confirmation message after removal

    # If train_button is pressed, proceed with training and saving the model
    if train_button:
        clean_data()  # Clean the data (e.g., handle missing values, preprocess)
        create_label()  # Create new labels for the data
        train_model()  # Train the model with the prepared data
        move_file()  # Move the trained model and files to the destination directory
        load_once = True  # Set load_once back to True to trigger data loading again when needed
        return "Trained Successfully"  # Return success message after training
