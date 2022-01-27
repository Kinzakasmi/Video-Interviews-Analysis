from audio_utils import pydub2librosa
import pydub
import librosa
import numpy as np

def pauses_features(audio) :    
    '''Returns min, max and mean of pauses and speak duration all in seconds'''
    # must be silent for at least 2 second
    # consider it silent if quieter than -40 dBFS
    silence   = pydub.silence.detect_silence(audio, min_silence_len=2000, silence_thresh=-40, seek_step=20)
    durations = [round((e-s)/1000) for (s,e) in silence]

    interview_duration = round(len(audio)/1000)

    if len(durations) == 0 :
        return 0,0,0,0,interview_duration
    else :
        speak_duration = interview_duration-np.sum(durations)
        durations      = [round(d/interview_duration,6) for d in durations] #normalization
        nb_long_pauses = round(len(durations)/interview_duration,6)
        mean_pauses    =  np.mean(durations)
        max_pauses     = np.max(durations)
        return nb_long_pauses, mean_pauses, max_pauses, speak_duration


def spectral_features(audio):
    '''Extracts spectral features from an audio, ie the means of the following :
    - Spectral centroid,
    - Spectral bandwidth,
    - Spectral rolloff,
    - Zero crossing rate,
    - Mel-Frequency Cepstral Coefficients(MFCCs),
    - Chroma features
    '''
    y, sr = pydub2librosa(audio)
    
    rmse = librosa.feature.rms(y=y)
    # Spectral centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    # Spectral bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    # Mel-Frequency Cepstral Coefficients(MFCCs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    # Chroma features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    to_append = [np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), np.mean(spec_bw),np.mean(rolloff),np.mean(zcr)]    
    for e in mfcc:
        to_append.append(np.mean(e))

    return to_append