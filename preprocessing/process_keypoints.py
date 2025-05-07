
import pandas as pd     # Import pandas library for handling and processing structured data (like DataFrames)
import os               # Import os module for interacting with the operating system (like file paths)
import time             # Import time module to handle time-related operations (e.g., delays, timestamps)
import warnings         # Import warnings module to manage or suppress warning messages

# ignore exactly the “feature names” UserWarning from sklearn.base
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but MinMaxScaler was fitted with feature names"
)

# List of 21 standard hand landmarks used in hand tracking models like MediaPipe Hands
# Each entry represents a specific joint or fingertip on the hand
hand_landmark_labels = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]


def capture_data(hand, VIDEO_WIDTH, VIDEO_HEIGHT, expires_at, Label=None):
    """
    Captures and processes hand landmark data from MediaPipe's hand model.
    Optionally labels and saves the data to a CSV file if within a time window.
    """
    
    # Dictionary to store the normalized landmark coordinates
    row = {}

    # Check if hand landmarks are detected
    if hand.multi_hand_landmarks:
        num_hands = len(hand.multi_hand_landmarks)

        # Process each detected hand
        for i, hand_landmarks in enumerate(hand.multi_hand_landmarks):
            # Use WRIST as the origin point for normalization
            origin_x = hand_landmarks.landmark[0].x * VIDEO_WIDTH
            origin_y = hand_landmarks.landmark[0].y * VIDEO_HEIGHT

            # Loop through each landmark and store relative x, y coordinates
            for name, landmark in zip(hand_landmark_labels, hand_landmarks.landmark):
                row[f'{i}_{name}_x'] = (landmark.x * VIDEO_WIDTH) - origin_x
                row[f'{i}_{name}_y'] = (landmark.y * VIDEO_HEIGHT) - origin_y

    # Get current time to check if data capture window is still valid
    now = time.time()

    # Save data only if:
    # - within the time window
    # - a label is provided
    # - enough landmarks are detected (at least 42 values)
    if (now < expires_at) and (Label is not None) and (len(row) >= 42):
        row['label'] = Label.upper()  # Add label to the data
        df = pd.DataFrame([row])  # Convert to DataFrame for CSV storage

        # Choose file path based on number of detected hands
        if num_hands == 1:
            csv_file = 'keypoints/1_custom_keypoint.csv'
        else:
            csv_file = 'keypoints/2_custom_keypoint.csv'

        # Write header only if the file doesn't already exist
        write_header = not os.path.exists(csv_file)

        # Append the row to the appropriate CSV file
        df.to_csv(csv_file, mode='a', header=write_header, index=False)

        # Return None to show that keypoint is stored
        return None

    # Return the row for further use (e.g., prediction)
    return row

def gen_result(row, model, normalizer, label_list):
    """
    Generates a classification result based on the processed keypoint row.
    Applies normalization and predicts using the appropriate model.
    """

    try:
        # Convert the row dictionary into a DataFrame for processing
        df = pd.DataFrame([row])

        # Determine if the input corresponds to a double-hand or single-hand model
        if len(df.columns) > 42:
            i = 1  # Double-hand
        else:
            i = 0  # Single-hand

        # Normalize the input keypoints
        normalized_value = normalizer[i].transform(df)

        # Predict the label index using the selected model
        res_idx = model[i].predict(normalized_value)[0]

        # Map the predicted index to the actual label string
        result = label_list[i][res_idx]

    except:
        # If any error occurs (e.g., shape mismatch, model/normalizer not ready), return default message
        result = 'Waiting for pose ....'

    return result