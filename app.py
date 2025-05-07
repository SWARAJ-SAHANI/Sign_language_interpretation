from flask import Flask, render_template, Response, request, redirect, url_for, jsonify     # Flask framework imports for web app functionality
import time                             # For timing operations or expiration checks
import threading                        # To handle concurrency (e.g., camera locking)
from joblib import load                 # Joblib for loading serialized ML models and normalizers
import pyttsx3                          # Text-to-speech conversion library

# Custom preprocessing modules for image handling and training logic
import preprocessing.image_processing as ip
import preprocessing.train_model as train

# Utility to load all pre-trained models and label encoders
from load_models import load_all_categories

# Load all pre-trained models, normalizers, and label lists for each category (e.g., single/double hand)
loaded_models = load_all_categories()

# Initialize the pyttsx3 engine for text-to-speech functionality
engine = pyttsx3.init()

# Variable to speak
say_text = True

# List of available sign language model categories for selection/display
model_category = ['numbers', 'alphabets', 'simple_signs', 'custom_signs']

# Default selected category is set to 'numbers'
selected_model_category = 'numbers'

# Load default model settings for "numbers"
model = loaded_models[selected_model_category]['models']
normalizer = loaded_models[selected_model_category]['normalizers']
label = loaded_models[selected_model_category]['labels']

# flask implementation
app = Flask(__name__)

# Home page route
@app.route('/')
def home():
    return render_template('home.html')

# Page showing supported gestures across different models
@app.route('/supported_gestures')
def supported_gestures():
    return render_template('supported_gestures.html')

# Live sign recognition using webcam and ML models
@app.route('/sign_recognition')
def sign_recognition():
    return render_template('sign_recognition.html')

# Custom sign recording and training interface
@app.route('/custom_sign')
def custom_sign():
    return render_template('custom_sign.html')

# Documentation or help section for users
@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

# Route to configure camera resolution and model category
@app.route('/configuration', methods=['GET', 'POST'])
def configuration():
    global model, normalizer, label, selected_model_category, is_custom_model_trained

    # Determine current resolution based on current width and height
    current_resolution = None
    for key, (w, h) in ip.RESOLUTIONS.items():
        if w == ip.VIDEO_WIDTH and h == ip.VIDEO_HEIGHT:
            current_resolution = key
            break

    # Handle POST request when the user submits the configuration form
    if request.method == 'POST':
        # Update resolution based on user selection
        new_res = request.form.get('resolution')
        if new_res in ip.RESOLUTIONS:
            ip.VIDEO_WIDTH, ip.VIDEO_HEIGHT = ip.RESOLUTIONS[new_res]
            current_resolution = new_res

        # Update the selected model category (e.g., NUMBERS, ALPHABETS, etc.)
        new_category = request.form.get('model_category')
        if new_category in model_category:
            selected_model_category = new_category
            model = loaded_models[new_category]['models']
            normalizer = loaded_models[new_category]['normalizers']
            label = loaded_models[new_category]['labels']

        # Redirect to avoid form resubmission issues on refresh
        return redirect(url_for('configuration'))

    # Render the configuration page with current settings
    return render_template(
        'configuration.html',
        current_resolution=current_resolution,
        resolutions=ip.RESOLUTIONS,
        model_category=model_category,
        selected_model_category=selected_model_category
    )

# Video stream endpoint for the webcam feed with real-time sign recognition
@app.route('/video_feed')
def video_feed():
    return Response(
        ip.gen_frames(model, normalizer, label),
        mimetype='multipart/x-mixed-replace; boundary=frame'  # Required for streaming MJPEG frames
    )

# Toggle the visibility of hand landmarks on the video feed (on/off switch via POST)
@app.route('/toggle_landmarks', methods=['POST'])
def toggle_landmarks():
    ip.show_landmarks = not ip.show_landmarks  # Flip the boolean flag
    print("SHOW_LANDMARKS is now", ip.show_landmarks)  # Log the current state
    return jsonify({"landmarks_enabled": ip.show_landmarks})  # Respond with the updated state

@app.route('/toggle_audio', methods=['POST'])
def toggle_audio():
    global say_text
    say_text = not say_text  # Flip the boolean flag
    print("text Audio is now", say_text)  # Log the current state
    return jsonify({"audio_enabled": say_text})  # Respond with the updated state

# SSE (Server-Sent Events) endpoint to stream real-time ML prediction text to the frontend
@app.route('/ml_stream')
def ml_stream():
    def generate():
        # This generator function continuously yields the current ML result.
        # This enables a live text display in the browser (like the recognized sign).
        while True:
            # Send the latest result as an SSE message.
            yield f"data: {ip.ml_result}\n\n"
            time.sleep(0.05)  # Throttle updates to ~20 times per second

    # Return the generator wrapped in a streaming response
    return Response(generate(), mimetype='text/event-stream')


@app.route('/speak', methods=['POST'])
def speak():
    data = request.get_json()
    text = data.get('text', '')
    if text:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    return jsonify({"status": "spoken", "text": text})
# @app.route('/speak', methods=['POST'])
# def speak():
#     if say_text:
#         engine = pyttsx3.init()
#         try:
#             data = request.get_json()
#             text = data.get('text', '')
#             if text:
#                 engine.say(text)
#                 engine.runAndWait()
#                 print('done')
#         except Exception as e:
#             print("audio error, ", e)
#         return jsonify({"status": "spoken", "text": text})

# Route for handling the custom model training interface
@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    action = None
    remove_index = None
    
    if request.method == 'POST':
        # Get the action type from the form submission: 'train' or 'remove'
        action = request.form.get('action')

        if action == 'remove':
            try:
                # Try to convert user input to integer for label removal
                remove_index = int(request.form.get('remove_index'))
            except (TypeError, ValueError):
                remove_index = None

            # Call training function with label removal (adjusted for zero-based index)
            result = train.main(train_button=False, remove_index=remove_index - 1)

        elif action == 'train':
            # Trigger the model training process
            result = train.main(train_button=True)
        
        else:
            # Default fallback if action is invalid
            result = train.main()

    else:
        # For GET request, just load the current state without any modifications
        result = train.main()
    
    # Decide whether to show labels in the UI or not
    if result == 'no Data' or result == 'Trained Successfully':
        labels = None  # No labels to show
    else:
        labels = train.labels  # Labels available to show/remove

    # Render the training page with current state and labels (if any)
    return render_template('train_model.html', result=result, labels=labels)


@app.route('/update_text', methods=['POST'])
def update_text():
    seconds =  10
    data = request.get_json()
    text = data.get("text", None)
    # Set the custom_text and its expiration (current time + 5 seconds).
    ip.custom_label = text
    ip.text_expires_at  = time.time() + seconds
    print("Custom text updated to:", text)
    return jsonify({"status": "success", "text": text})


if __name__ == '__main__':
    app.run(debug=True)