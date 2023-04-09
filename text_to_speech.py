def text_to_speech(api_key, region, text, output_file_path):
    from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig, SpeechSynthesisOutputFormat, ResultReason
    from azure.cognitiveservices.speech.audio import AudioOutputConfig

    speech_config = SpeechConfig(subscription=api_key, region=region)
    speech_config.set_speech_synthesis_output_format(SpeechSynthesisOutputFormat["Riff16Khz16BitMonoPcm"])

    # Use a neural voice instead of a standard voice
    speech_config.speech_synthesis_voice_name = "en-US-JessaNeural"

    audio_output_config = AudioOutputConfig(filename=output_file_path)
    synthesizer = SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)

    result = synthesizer.speak_text_async(text).get()

    if result.reason == ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print(f"Speech synthesis canceled: {cancellation_details.reason}")
        print(f"Error details: {cancellation_details.error_details}")
    else:
        print("Text to speech conversion successful!")