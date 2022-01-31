import speech_recognition as sr 


def speech_recognition(filename):
    """
    Listen from preprocessed audio (ie silences has to be removed for Recognize to work) to
    compute speech as a string
    """
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_file = r.record(source)
    return r.recognize_google(audio_file,language = 'fr-FR', show_all=False)
