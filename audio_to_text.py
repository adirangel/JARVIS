import requests
import audioop

# audio_to_text.py
def audio_to_text(api_key, audio_file_path, silence_threshold=200):
    # Check if the audio is silent or too quiet
    with open(audio_file_path, "rb") as audio_file:
        audio_data = audio_file.read()
        rms = audioop.rms(audio_data, 2)

    if rms < silence_threshold:
        print("Silent or too quiet audio detected, stopping.")
        return ""

    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "model": "whisper-1",
    }

    with open(audio_file_path, "rb") as audio_file:
        files = {
            "file": (audio_file_path, audio_file, "audio/wav"),
        }
        response = requests.post("https://api.openai.com/v1/audio/transcriptions", headers=headers, data=data, files=files)

    response_json = response.json()
    print("Response JSON:", response_json)  # Print the full response for debugging

    try:
        transcribed_text = response_json['text']
    except KeyError:
        transcribed_text = "Error: Unable to extract transcribed text from the API response."

    # Check if the transcribed text consists only of periods or spaces
    if all(c in {'.', ' '} for c in transcribed_text):
        print("No meaningful speech detected, stopping.")
        return ""

    return transcribed_text
