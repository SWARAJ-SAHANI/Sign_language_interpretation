import pyttsx3
def say(text):
    engine = pyttsx3.init()
    print("say the ---> ", text)
    engine.say(text)
    engine.runAndWait()
    print(f"says text: {text}")
    return

