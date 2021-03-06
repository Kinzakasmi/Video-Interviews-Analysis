import pydub
import librosa
import numpy as np
import itertools
import pandas as pd

#Only for praat
import os
import parselmouth 
from parselmouth import praat

import math

# ---- Read/Split audio ----

def pydub2librosa(audio):
    '''Converts pydub audio format to librosa audio format
    Returns a tuple of (list,int) corresponding to the librosa audio and the frame rate.'''
    audio = audio.set_channels(1) # to mono audio
    y = audio.get_array_of_samples()
    y = librosa.util.buf_to_float(y,n_bytes=2,dtype=np.float32)
    return y, audio.frame_rate


def split_questions(video_folder,df_startend,filename):
    '''Splits a video into several videos given a frame of start and end.
    Arguments:
        video_folder: str. The folder containing videos.
        df_startend: Pandas DataFrame. DataFrame containing columns 'email', 'start' and 'end' for each question. 
        filename: str. The title of the video file.
    Returns a list of chunk videos extracted from a given video interview.
    '''
    df_startend = df_startend[df_startend['email']==filename.split('.mp4',2)[0]]
    #Read audio
    audio = pydub.AudioSegment.from_file(video_folder+filename,'mp4')
    #split audio
    audios = [audio[s:e] for (s,e) in zip(df_startend['start'],df_startend['end'])]
    return audios


# ---- Pause related features ----
def pauses_features(silent_ranges,length) :    
    """Attributes the min, max and mean of pauses in seconds
    Arguments:
        silent_ranges: list. A list of (start,end) of silent ranges.
        length: int. The length of the input audio
    Returns a Pandas DataFrame of statistics related to pauses."""
    silence_durations = [round((e-s)/1000) for (s,e) in silent_ranges]

    if len(silence_durations) == 0 :
        nb_long_pauses = 0
        mean_pauses    = 0
        max_pauses     = 0
    else :
        #Compute statistics
        durations      = [round(d/length,6) for d in silence_durations] #normalization
        nb_long_pauses = round(len(np.array(durations)>5)/length,6)
        mean_pauses    = np.mean(durations)
        max_pauses     = np.max(durations)

    df_features       = pd.DataFrame([nb_long_pauses, mean_pauses, max_pauses],index=['nb_pauses','mean_pauses','max_pauses'])
    return df_features.transpose()


def nonsilent_ranges(silent_ranges,length):
    """Attributes a list of all NONsilent sections of an audio.
    Arguments:
        silent_ranges: list. A list of (start,end) of silent ranges.
        length: int. The length of the input audio
    Returns a list (start,end) (in ms) of non silent ranges"""
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
    """Removes silence from audio.
    Arguments: 
        audio: pydub.AudioSegment object. The audio to be processed.
        nonsilent_ranges: list. Non silent ranges of the audio.
        keep_silence: int. keep_silence - (in ms or True/False) leave some silence at the beginning
            and end of the chunks. Keeps the sound from sounding like it is abruptly cut off.
            When the length of the silence is less than the keep_silence duration
            it is split evenly between the preceding and following non-silent segments.
            If True is specified, all the silence is kept, if False none is kept.
            default: 1000ms
    Returns pydub.AudioSegment object without silence and its new length."""
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
            range_i[1]  = (last_end+next_start)//2
            range_ii[0] = range_i[1]

    audios = [audio[ max(start,0) : min(end,len(audio)) ] for start,end in output_ranges]
    
    audio = audios[0]
    for a in audios[1:]:
        audio = audio.append(a,crossfade=0)                

    length = round(len(audio)/1000)
    return audio, length


# ---- Spectral features ----
def spectral_features(y, sr, n_fft, hop_length):
    """Extracts Mel-Frequency Cepstral Coefficients(MFCCs)
    Arguments: 
        y: np.ndarray [shape=(..., n,)]. Audio time series. 
        sr: number > 0 [scalar]. audio sampling rate of ``y``.
        n_fft : int > 0 [scalar]. FFT window size
        hop_length : int > 0 [scalar]. hop length for STFT. See `librosa.stft` for details.
    Returns a Pandas DataFrame object containing statistics related to MFCCs.
    """
    ## You can un-comment the following if you want to use spectral features. But our exploratory analysis showed
    ## that only the MFCCs are relevant to score interview performances.

    #rmse      = librosa.feature.rms(y=y, hop_length=hop_length)
    #spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    #spec_bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    #rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    #zcr       = librosa.feature.zero_crossing_rate(y)

    # Mel-Frequency Cepstral Coefficients(MFCCs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

    # Formatting the DataFrame
    feature_names     = ['mfcc'+str(i) for i in range(20)]

    df_features       = pd.DataFrame(mfcc,index=feature_names)
    df_features       = df_features.transpose()
    #Compute statistics
    df_features       = df_features.apply(lambda x : [np.nanmean(x),np.nanstd(x),np.nanmin(x),np.nanmax(x)],axis=0)
    df_features.index = ['mean','std','min','max']
    
    return df_features


def loudness_features(y,sr,n_fft,hop_length):
    """The definition of loudness is very complex. This is just a try at finding loudness.
    Returns the mean, std, min and max of loudness
    Arguments:
        y: np.ndarray [shape=(..., n,)]. Audio time series. 
        sr: number > 0 [scalar]. audio sampling rate of ``y``.
        n_fft : int > 0 [scalar]. FFT window size
        hop_length : int > 0 [scalar]. hop length for STFT. See `librosa.stft` for details.
    Returns a Pandas DataFrame object containing statistics related to loudness and PSD.
    """    
    #Compute fft
    S     = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    # Compute power.
    power = np.abs(S)**2
    
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    a_weighting = librosa.A_weighting(frequencies)
    weighting   = 10**(a_weighting/10)
    power       = power.T * weighting

    power    = np.mean(power, axis=-1)
    loudness = np.log(power*100 + 1)

    # Formatting the DataFrame
    df_features       = pd.DataFrame([loudness,power],index=['loudness','psd'])
    df_features       = df_features.transpose()
    df_features       = df_features.apply(lambda x : [np.nanmean(x),np.nanstd(x),np.nanmin(x),np.nanmax(x)],axis=0)
    df_features.index = ['mean','std','min','max']
    
    return df_features


def f0_features(sound,f0min,f0max):
    """Returns the mean, std, min and max of f0 : time series of fundamental frequencies in Hertz.
    Arguments:
        sound. parselmouth.Sound object. The audio to be processed.
        f0min: int. The f0 min.
        f0max: int. The f0 max.
    """
    pitch = praat.call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    duration = praat.call(pitch, "Get end time")
    meanF0   = praat.call(pitch, "Get mean", 0, duration, "Hertz") # get mean pitch
    stdevF0  = praat.call(pitch, "Get standard deviation", 0 , duration, "Hertz") # get standard deviation
    minF0    = praat.call(pitch, "Get minimum", 0, duration, "Hertz", "Parabolic")
    maxF0    = praat.call(pitch, "Get maximum", 0, duration, "Hertz", "Parabolic")
    
    # Formatting the DataFrame
    df_features       = pd.DataFrame([meanF0,stdevF0,minF0,maxF0],index=['mean','std','min','max'],columns=['f0'])    
    return df_features


def get_formants(sound,f0min,f0max):
    """Extracts formants (frequency peaks in the spectrum which have a high degree of energy).
    This functions uses Praat software. I will look into implementing this myself.
    Arguments:
        sound. parselmouth.Sound object. The audio to be processed.
        f0min: int. The f0 min.
        f0max: int. The f0 max.
    Returns a Pandas DataFrame of statistics related to the first 3 formants f1, f2 and f3.
    """
    pointProcess = praat.call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    formants = praat.call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)

    numPoints = praat.call(pointProcess, "Get number of points")
    f1_list = []
    f2_list = []
    f3_list = []
    for point in range(0, numPoints):
        point += 1
        t  = praat.call(pointProcess, "Get time from index", point)
        f1 = praat.call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = praat.call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = praat.call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)

    # Formatting the DataFrame
    df_features       = pd.DataFrame([f1_list,f2_list,f3_list],index=['f1','f2','f3'])
    df_features       = df_features.transpose()
    df_features       = df_features.apply(lambda x : [np.nanmean(x),np.nanstd(x),np.nanmin(x),np.nanmax(x)],axis=0)
    df_features.index = ['mean','std','min','max']
    return df_features


def speech_rate(sound):
    """Extracts speaking rate and other prosodic features
    Arguments:
        sound: parselmouth.Sound object. The input audio.
    Returns: a Pandas DataFrame of prosodic features containing the following columns:
        (voicedcount, npause, originaldur, intensity_duration, speakingrate, articulationrate, asd)
    """
    silencedb = -20 #relative to maximum
    mindip = 2
    minpause = 0.3

    originaldur = sound.get_total_duration()
    duration = praat.call(sound, "Get end time")
    intensity = sound.to_intensity(50)
    start = praat.call(intensity, "Get time from frame number", 1)
    nframes = praat.call(intensity, "Get number of frames")
    end = praat.call(intensity, "Get time from frame number", nframes)
    min_intensity = praat.call(intensity, "Get minimum", 0, duration, "Parabolic")
    max_intensity = praat.call(intensity, "Get maximum", 0, duration, "Parabolic")

    # get .99 quantile to get maximum (without influence of non-speech sound bursts)
    max_99_intensity = praat.call(intensity, "Get quantile", 0, 0, 0.99)

    # estimate Intensity threshold
    threshold = max_99_intensity + silencedb
    threshold2 = max_intensity - max_99_intensity
    threshold3 = silencedb - threshold2

    if threshold < min_intensity:
        threshold = min_intensity

    # get pauses (silences) and speakingtime
    textgrid = praat.call(intensity, "To TextGrid (silences)", threshold3, minpause, 0.1, "silent", "sounding")
    silencetier = praat.call(textgrid, "Extract tier", 1)
    silencetable = praat.call(silencetier, "Down to TableOfReal", "sounding")
    
    npauses = praat.call(silencetable, "Get number of rows")
    speakingtot = 0
    for ipause in range(npauses):
        pause = ipause + 1
        beginsound = praat.call(silencetable, "Get value", pause, 1)
        endsound = praat.call(silencetable, "Get value", pause, 2)
        speakingdur = endsound - beginsound
        speakingtot += speakingdur


    intensity_matrix = praat.call(intensity, "Down to Matrix")
    # sndintid = sound_from_intensity_matrix
    sound_from_intensity_matrix = praat.call(intensity_matrix, "To Sound (slice)", 1)
    # use total duration, not end time, to find out duration of intdur (intensity_duration)
    # in order to allow nonzero starting times.
    intensity_duration = praat.call(sound_from_intensity_matrix, "Get total duration")
    intensity_max = praat.call(sound_from_intensity_matrix, "Get maximum", 0, 0, "Parabolic")
    point_process = praat.call(sound_from_intensity_matrix, "To PointProcess (extrema)", "Left", "yes", "no", "Sinc70")
    # estimate peak positions (all peaks)
    numpeaks = praat.call(point_process, "Get number of points")
    t = [praat.call(point_process, "Get time from index", i + 1) for i in range(numpeaks)]

    # fill array with intensity values
    timepeaks = []
    peakcount = 0
    intensities = []
    for i in range(numpeaks):
        value = praat.call(sound_from_intensity_matrix, "Get value at time", t[i], "Cubic")
        if value > threshold:
            peakcount += 1
            intensities.append(value)
            timepeaks.append(t[i])

    # fill array with valid peaks: only intensity values if preceding
    # dip in intensity is greater than mindip
    validpeakcount = 0
    currenttime = timepeaks[0]
    currentint = intensities[0]
    validtime = []
    for p in range(peakcount - 1):
        following = p + 1
        followingtime = timepeaks[p + 1]
        dip = praat.call(intensity, "Get minimum", currenttime, timepeaks[p + 1], "None")
        
        
        diffint = abs(currentint - dip)
        if diffint > mindip:
            validpeakcount += 1
            validtime.append(timepeaks[p])
        else:
            currenttime = timepeaks[following]
            currentint = praat.call(intensity, "Get value at time", timepeaks[following], "Cubic")
    
    # Look for only voiced parts
    pitch = sound.to_pitch_ac(0.02, 30, 4, False, 0.03, 0.25, 0.01, 0.35, 0.25, 450)
    voicedcount = 0
    voicedpeak = []
    timecorrection = originaldur / intensity_duration
    for time in range(validpeakcount):
        querytime = validtime[time]
        whichinterval = praat.call(textgrid, "Get interval at time", 1, querytime)
        whichlabel = praat.call(textgrid, "Get label of interval", 1, whichinterval)
        value = pitch.get_value_at_time(querytime) 
        if not math.isnan(value):
            if whichlabel == "sounding":
                voicedcount += 1
                voicedpeak.append(validtime[time])

    # calculate time correction due to shift in time for Sound object versus
    # intensity object

    # Insert voiced peaks in TextGrid
    praat.call(textgrid, "Insert point tier", 1, "syllables")
    for i in range(len(voicedpeak)):
        position = (voicedpeak[i] * timecorrection)
        praat.call(textgrid, "Insert point", 1, position, "")

    # return results
    speakingrate = voicedcount / originaldur
    articulationrate = voicedcount / speakingtot
    npause = npauses - 1
    asd = speakingtot / voicedcount

    df_features = pd.DataFrame([voicedcount, npause, originaldur, intensity_duration, speakingrate, articulationrate, asd],
                            index=['voicedcount', 'npause', 'originaldur', 'intensity_duration', 'speakingrate', 
                                    'articulationrate', 'asd'])
    df_features = df_features.transpose()
    return df_features


def prosodic_features(audio,n_fft=2048,hop_length=512,f0min=75,f0max=300):
    """Extracts prosodic features. They capture the intonation of speech, the rhythm or the tone of speech. 
    They reveal the information about the identity, attitude and emotional state of the underlying signal.
    Arguments:
        audio: pydub.AudioSegment. The input audio
        n_fft : int > 0 [scalar]. FFT window size
        hop_length : int > 0 [scalar]. hop length for STFT. See `librosa.stft` for details.
        f0min: int > 0. The f0 min.
        f0max: int > 0. The f0 max.
    Returns a Pandas DataFrame of several prosodic features and their statistics.
    """
    #Convert to a librosa audio format.
    y, sr = pydub2librosa(audio)

    #Get spectral features
    spectral_feats = spectral_features(y, sr,n_fft=n_fft,hop_length=hop_length)
    
    #Get loudness
    loudness_feats = loudness_features(y,sr,n_fft,hop_length)
    
    ## Using Praat software
    audio.export('temp.wav',format ="wav") #No other option that exporting in .wav temporarily
    tempfile = os.getcwd()+"/temp.wav"
    sound = parselmouth.Sound(tempfile) 

    # Get f0
    f0_feats = f0_features(sound,f0min,f0max)

    # Get formants 
    formants_feats = get_formants(sound,f0min,f0max)

    # Get speech info
    speech_feats = speech_rate(sound)
    os.remove(tempfile)
    return pd.concat([spectral_feats, f0_feats, loudness_feats, formants_feats], axis=1, join="inner"), speech_feats


class Audio :
    """The Audio Class.
    Argments:
        audio: pudub.AudioSegment object. The input audio.
        title: str. The title of the file audio (the same title can be found in the scoring sheet as well as the 
            mp4 audio file.
        question: int. The question number.
        min_silence_len: int. The minimum length for any silent section. Default at 2000ms.
        silence_thresh: int. The upper bound for how quiet is silent in dFBS (in comparison with the maximum). 
            Default at -30dB.
        keep_silence: int in ms or True/False. Leave some silence at the beginning and end of the chunks. 
            When the length of the silence is less than the keep_silence duration, it is split evenly between the 
            preceding and following non-silent segments.
            If True is specified, all the silence is kept, if False none is kept.
            Default at 1000ms.
        n_fft : int > 0 [scalar]. FFT window size
        hop_length : int > 0 [scalar]. hop length for STFT. See `librosa.stft` for details.
        """
    def __init__(self,audio,title,question,min_silence_len=2000,silence_thresh=-30,keep_silence=1000,
                n_fft=2048,hop_length=512):
        self.audio    = audio
        self.email    = title
        self.question = question
        self.length   = round(len(self.audio)/1000)
       
        self.silent_ranges     = []
        self.nonsilent_ranges  = []
        self.spectral_features = []
        self.prosodic_features = []

        self.min_silence_len = min_silence_len
        self.silence_thresh  = silence_thresh
        self.keep_silence    = keep_silence
        self.n_fft           = n_fft
        self.hop_length      = hop_length

    def preprocessing(self):
        # Detecting silent parts
        self.silence_thresh = pydub.utils.ratio_to_db(self.audio.max/self.audio.max_possible_amplitude) + self.silence_thresh
        self.silent_ranges = pydub.silence.detect_silence(self.audio, 
                                                        min_silence_len=self.min_silence_len, 
                                                        silence_thresh=self.silence_thresh, 
                                                        seek_step=20)
        # Calculating pause-related features
        pauses_feats = pauses_features(self.silent_ranges,self.length)
        
        # Getting nonsilent parts
        self.nonsilent_ranges = nonsilent_ranges(self.silent_ranges,self.length)
        
        # Removing silent parts from audio
        self.audio, self.length = split_non_silent(self.audio,self.nonsilent_ranges,keep_silence=self.keep_silence)        

        # Calculating prosodic_features
        self.spectral_features, self.prosodic_features = prosodic_features(self.audio,n_fft=self.n_fft,hop_length=self.hop_length)
        self.prosodic_features = pd.concat([self.prosodic_features,pauses_feats],axis=1)