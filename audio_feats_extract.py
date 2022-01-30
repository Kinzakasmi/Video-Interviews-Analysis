from imp import load_dynamic
from pyexpat import features
from audio_utils import pydub2librosa
import pydub
import librosa
import numpy as np
import itertools
import pandas as pd
import os
import parselmouth 
from parselmouth import praat

# ---- Pause related features ----
def pauses_features(silent_ranges,length) :    
    """Attributes the min, max and mean of pauses in seconds"""
    silence_durations = [round((e-s)/1000) for (s,e) in silent_ranges]

    if len(silence_durations) == 0 :
        return [0,0,0,0], ['']
    else :
        durations      = [round(d/length,6) for d in silence_durations] #normalization
        nb_long_pauses = round(len(durations)/length,6)
        mean_pauses    =  np.mean(durations)
        max_pauses     = np.max(durations)
        return [nb_long_pauses, mean_pauses, max_pauses]

def nonsilent_ranges(silent_ranges,length):
    """Attributes a list of all nonsilent sections of an audio."""
    len_seg = length*1000 #in ms

    # if there is no silence, the whole thing is nonsilent
    if silent_ranges==[]:
        return [[0, len_seg]]

    # short circuit when the whole audio segment is silent
    if silent_ranges[0][0] == 0 and silent_ranges[0][1] == len_seg:
        return []

    prev_end_i = 0
    nonsilent_ranges = []
    for start_i, end_i in silent_ranges:
        nonsilent_ranges.append([prev_end_i, start_i])
        prev_end_i = end_i

    if end_i != len_seg:
        nonsilent_ranges.append([prev_end_i, len_seg])

    if nonsilent_ranges[0] == [0, 0]:
        nonsilent_ranges.pop(0)

    return nonsilent_ranges

def split_non_silent(audio, nonsilent_ranges,keep_silence=1000):
    """Attributes list of audio segments from splitting audio_segment on silent sections"""
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    if isinstance(keep_silence, bool):
        keep_silence = len(audio) if keep_silence else 0

    output_ranges = [
        [ start - keep_silence, end + keep_silence ] for (start,end) in nonsilent_ranges
    ]

    for range_i, range_ii in pairwise(output_ranges):
        last_end = range_i[1]
        next_start = range_ii[0]
        if next_start < last_end:
            range_i[1] = (last_end+next_start)//2
            range_ii[0] = range_i[1]

    audios = [audio[ max(start,0) : min(end,len(audio)) ] for start,end in output_ranges]
    
    audio = audios[0]
    for a in audios[1:]:
        audio = audio.append(a,crossfade=0)                

    length = round(len(audio)/1000)
    return audio, length

# ---- Spectral features ----
def spectral_features(audio,n_fft, hop_length):
    """Extracts spectral features from an audio, ie the means of the following :
    - Spectral centroid,
    - Spectral bandwidth,
    - Spectral rolloff,
    - Zero crossing rate,
    - Mel-Frequency Cepstral Coefficients(MFCCs),
    - Chroma features"""
    y, sr = pydub2librosa(audio)
    
    rmse = librosa.feature.rms(y=y, hop_length=hop_length)
    # Spectral centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # Spectral bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    # Mel-Frequency Cepstral Coefficients(MFCCs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    # Chroma features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

    features = [rmse[0,:], spec_cent[0,:], spec_bw[0,:], rolloff[0,:], zcr[0,:]]   
    for c in chroma_stft :
        features.append(c)

    for e in mfcc:
        features.append(e)    

    feature_names = ['rms','spec_cent','specbw','rolloff','zcr']+['chroma_stft'+str(i) for i in range(12)]+['mfcc'+str(i) for i in range(20)]

    df_features = pd.DataFrame(features,index=feature_names)
    df_features = df_features.transpose()

    df_features = df_features.apply(lambda x : [np.mean(x),np.std(x),np.min(x),np.max(x)],axis=0)
    df_features.index = ['mean','std','min','max']
    return df_features

# ---- Prosodic features ----
def f0_features(y,sr):
    """Returns the mean, std, min and max of :
        - f0 : time series of fundamental frequencies in Hertz.
        - voiced_prob : time series containing the probability that a frame is voiced.
    """
    f0, _, voiced_prob = librosa.pyin(y, sr=sr, frame_length=2048*8,
                            fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    df_features = pd.DataFrame([f0,voiced_prob],index=['f0','voiced_pb'])
    df_features = df_features.transpose()

    df_features = df_features.apply(lambda x : [np.mean(x),np.std(x),np.min(x),np.max(x)],axis=0)
    df_features.index = ['mean','std','min','max']
    return df_features

def tempo_features(y,sr,hop_length):
    """Returns the estimated tempo (beats per minute)"""
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length,aggregate=None)

    df_features = pd.DataFrame(tempo,columns=['tempo'])

    df_features = df_features.apply(lambda x : [np.mean(x),np.std(x),np.min(x),np.max(x)],axis=0)
    df_features.index = ['mean','std','min','max']
    return df_features

def loudness_features(y,sr,n_fft,hop_length):
    """The definition of loudness is very complex. This is just a try at finding loudness.
    Returns the mean, std, min and max of loudness"""    
    #Compute fft
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    # Compute power.
    power = np.abs(S)**2
    
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    a_weighting = librosa.A_weighting(frequencies)
    weighting = 10**(a_weighting/10)
    power = power.T * weighting

    power = np.mean(power, axis=-1)
    loudness = np.log(power*100 + 1)

    df_features = pd.DataFrame([loudness,power],index=['loudness','psd'])
    df_features = df_features.transpose()

    df_features = df_features.apply(lambda x : [np.mean(x),np.std(x),np.min(x),np.max(x)],axis=0)
    df_features.index = ['mean','std','min','max']
    return df_features

def get_formants(audio):
    """Extracts formants (frequency peaks in the spectrum which have a high degree of energy).
    This functions uses Praat software. I will look into implementing this myself."""

    #No other option that exporting in .wav temporarily
    audio.export('test.wav')
    tempfile = os.getcwd()+"/test.wav"
    sound = parselmouth.Sound(tempfile) 

    f0min=75
    f0max=300
    pointProcess = praat.call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    formants = praat.call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)

    numPoints = praat.call(pointProcess, "Get number of points")
    f1_list = []
    f2_list = []
    f3_list = []
    for point in range(0, numPoints):
        point += 1
        t = praat.call(pointProcess, "Get time from index", point)
        f1 = praat.call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = praat.call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = praat.call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)

    df_features = pd.DataFrame([f1_list,f2_list,f3_list],index=['f1','f2','f3'])
    df_features = df_features.transpose()

    df_features = df_features.apply(lambda x : [np.mean(x),np.std(x),np.min(x),np.max(x)],axis=0)
    df_features.index = ['mean','std','min','max']
    
    os.remove(tempfile)
    return df_features
    

def prosodic_features(audio,n_fft=2048,hop_length=512):
    """Extracts prosodic features. They capture the intonation of speech, the rhythm or the tone of speech. 
    They reveal the information about the identity, attitude and emotional state of the underlying signal.
    """
    y, sr = pydub2librosa(audio)
    # f0
    f0_feats = f0_features(y,sr)
    #tempo
    tempo_feats = tempo_features(y,sr,hop_length)
    #loudness
    loudness_feats = loudness_features(y,sr,n_fft,hop_length)
    #formants
    formants_feats = get_formants(audio)

    return pd.concat([f0_feats, tempo_feats, loudness_feats, formants_feats], axis=1, join="inner")

class Audio :
    def __init__(self,audio,min_silence_len=2000,silence_thresh=-40,keep_silence=1000,n_fft=2048,hop_length=512):
        self.audio = audio
        self.length = round(len(self.audio)/1000)
        self.silent_ranges = []
        self.pauses_features = []
        self.nonsilent_ranges = []
        self.spectral_features = []
        self.prosodic_features = []
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh
        self.keep_silence = keep_silence
        self.n_fft = n_fft
        self.hop_length = hop_length

    def preprocessing(self):
        # Detecting silent parts
        self.silent_ranges = pydub.silence.detect_silence(self.audio, 
                                                        min_silence_len=self.min_silence_len, 
                                                        silence_thresh=self.silence_thresh, 
                                                        seek_step=20)
        # Calculating pause-related features
        self.pauses_features = pauses_features(self.silent_ranges,self.length)
        
        # Getting nonsilent parts
        self.nonsilent_ranges = nonsilent_ranges(self.silent_ranges,self.length)
        
        # Removing silent parts from audio
        self.audio, self.length = split_non_silent(self.audio,self.nonsilent_ranges,keep_silence=self.keep_silence)

        # Calculating spectral features
        self.spectral_features = spectral_features(self.audio,n_fft=self.n_fft,hop_length=self.hop_length)

        # Calculating prosodic_features
        self.prosodic_features = prosodic_features(self.audio,n_fft=self.n_fft,hop_length=self.hop_length)