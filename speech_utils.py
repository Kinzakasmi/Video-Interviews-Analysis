import speech_recognition as sr 
import numpy as np


def speech_recognition(filename):
    """
    Listen from preprocessed audio (ie silences has to be removed for Recognize to work) to
    compute speech as a string
    """
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_file = r.record(source)
    return r.recognize_google(audio_file,language = 'fr-FR', show_all=False)



def stats(L_floats):
    """
    Returns stats from float list.
    """

    dico = {
        'mean'   : np.mean(L_floats), 
        'median' : np.median(L_floats), 
        'std'    : np.std(L_floats), 
        '95c'    : np.percentile(L_floats, 95), 
        'max'    : np.max(L_floats)
    }
        
    return dico
