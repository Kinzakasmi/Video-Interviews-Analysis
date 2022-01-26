from audio_utils import pydub2librosa
import pydub
import librosa
import numpy as np

def pauses_features(audio) :    
    '''Returns min, max and mean of pauses and speak duration all in seconds'''
    # must be silent for at least 1.5 second
    # consider it silent if quieter than -40 dBFS
    silence = pydub.silence.detect_silence(audio, min_silence_len=1500, silence_thresh=-40, seek_step=10)
    durations = [round((e-s)/1000) for (s,e) in silence]

    speak_duration = round(len(audio)/1000)-np.sum(durations)
    if len(durations) == 0 :
        return None
    else :
        return len(durations), np.min(durations), np.max(durations), np.mean(durations), speak_duration


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