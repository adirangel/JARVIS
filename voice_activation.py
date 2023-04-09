import speech_recognition as sr
import time

def listen_for_activation_phrase(activation_phrase, timeout_seconds=5):
    recognizer = sr.Recognizer()
    start_time = time.time()

    with sr.Microphone() as source:
        print("Listening for the activation phrase...")
        while True:
            try:
                audio = recognizer.listen(source, timeout=1)
                text = recognizer.recognize_google(audio)
                if text.lower() == activation_phrase.lower():
                    print(f"Activation phrase '{activation_phrase}' detected.")
                    return True
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                print("Error: Unable to connect to the speech recognition service.")
                return False
            except sr.WaitTimeoutError:
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout_seconds:
                    print(f"Timeout reached: {timeout_seconds} seconds have passed without detecting the activation phrase.")
                    return False
