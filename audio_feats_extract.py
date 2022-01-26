import pydub
import numpy as np

def pauses_features(audio) :    
    '''Returns min, max and mean of pauses and speak duration all in seconds'''
    # must be silent for at least 1.5 second
    # consider it silent if quieter than -40 dBFS
    silence = pydub.silence.detect_silence(audio, min_silence_len=1500, silence_thresh=-40, seek_step=10)
    durations = [round((e-s)/1000) for (s,e) in silence]

    speak_duration = round(len(audio)/1000)-np.sum(durations)
    return np.min(durations), np.max(durations), np.mean(durations), speak_duration