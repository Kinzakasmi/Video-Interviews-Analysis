import speech_recognition as sr 
import numpy as np
import pandas as pd



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
        'min'    : np.min(L_floats),
        'mean'   : np.mean(L_floats), 
        'median' : np.median(L_floats), 
        'std'    : np.std(L_floats), 
        '95c'    : np.percentile(L_floats, 95), 
        'max'    : np.max(L_floats)
    }
        
    return dico

def getDictionary(filename = 'http://www.lexique.org/databases/Lexique383/Lexique383.tsv'):
    lex = pd.read_csv(filename, sep='\t')
    return lex[["ortho", "lemme", "cgram", "freqfilms2", "freqlivres", "nblettres", "nbphons", "nbsyll"]]
