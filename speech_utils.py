import speech_recognition as sr 

def speech_recognition(filename):
    r = sr.Recognizer()
    audio = sr.AudioFile(filename)
    with audio as source:
        audio_file = r.record(source)
    
    result = r.recognize_google(audio_file,language = 'fr-FR', show_all=False)
    return result