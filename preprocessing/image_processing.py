import cv2                                          # mainly used for real-time computer vision tasks
import mediapipe as mp                              # useful for detecting hand landmarks, pose, face, etc.
import sys                                          # used for accessing system-specific parameters and functions
import threading                                    # allows running multiple threads (useful for handling video capture and processing in parallel)
import preprocessing.process_keypoints as kp        # Import custom keypoint processing functions from the preprocessing package

# Create a lock object to manage access to the camera resource across multiple threads
camera_lock = threading.Lock()

# Default video dimensions (can be used for webcam capture)
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# Predefined resolution settings for easy switching
RESOLUTIONS = {
    "480p": (640, 480),     # Standard definition
    "720p": (1280, 720),    # HD resolution
    "1080p": (1920, 1080)   # Full HD resolution
}

# Utility module from MediaPipe for drawing landmarks and connections on images
mp_drawing = mp.solutions.drawing_utils

# Hands module from MediaPipe for detecting and tracking hand landmarks
mp_hands = mp.solutions.hands

# Initialize MediaPipe Hands model with confidence thresholds
hands_model = mp_hands.Hands(
    min_detection_confidence=0.5,  # Minimum confidence for detecting a new hand
    min_tracking_confidence=0.5    # Minimum confidence for tracking an already detected hand
)

# Flag to control whether to draw hand landmarks on the video feed
show_landmarks = True

# Default message displayed while waiting for hand detection or model prediction
ml_result = 'Waiting for pose ....'

# Placeholder for a custom label (e.g., for manual annotation or override)
custom_label = None

# Timestamp to determine how long a label or message should be displayed
text_expires_at = 0


# Processes a single video frame:
# - Applies hand and pose detection
# - Optionally classifies the gesture
# - Draws landmarks if enabled
# - Returns the annotated frame
def process_frame(frame, model, norm, label_list):

    global ml_result

    # Flip the frame horizontally for a mirror-like effect (more intuitive for user)
    frame = cv2.flip(frame, 1)

    # Convert the frame from BGR (OpenCV default) to RGB (MediaPipe expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks with MediaPipe Hands
    results = hands_model.process(rgb_frame)

    # Capture keypoints: if custom_label is set, include it (used for data collection or annotation)
    if custom_label is not None:
        row = kp.capture_data(results, VIDEO_WIDTH, VIDEO_HEIGHT, text_expires_at, custom_label)
    else:
        row = kp.capture_data(results, VIDEO_WIDTH, VIDEO_HEIGHT, text_expires_at)

        # If keypoints are captured successfully, run prediction
        ml_result = kp.gen_result(row, model, norm, label_list)

    # If landmark display is enabled and hands were detected, draw them on the frame
    if show_landmarks and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Return the processed (and potentially annotated) frame
    return frame

# Generator function that captures video frames, processes them, and yields JPEG-encoded frames.
# Used for streaming real-time hand gesture predictions.
def gen_frames(model, norm, label_list):

    # Set initial machine learning result message
    global ml_result
    ml_result = 'Waiting for pose ....'

    # Lock camera access to prevent race conditions in multi-threaded environments
    with camera_lock:

        # Use appropriate camera backend based on operating system
        if sys.platform.startswith('win'):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # For Windows (avoids webcam lag)
        else:
            cap = cv2.VideoCapture(0)  # Default for Linux/macOS

        # Set video frame dimensions
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

        try:
            while True:
                # Read a frame from the webcam
                success, frame = cap.read()
                if not success:
                    print("Failed to grab frame")
                    break

                # Process the frame (detect hands, predict gesture, draw results)
                frame = process_frame(frame, model, norm, label_list)

                # Encode the frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue

                # Yield the frame in a format suitable for MJPEG streaming
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        finally:
            # Always release the camera resource when done
            cap.release()
            print("Camera released.")
