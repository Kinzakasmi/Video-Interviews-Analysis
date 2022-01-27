from audio_utils import pydub2librosa
import pydub
import librosa
import numpy as np
import itertools

class Audio :
    def __init__(self,audio):
        self.audio = audio
        self.length = round(len(self.audio)/1000)
        self.silent_ranges = []
        self.pauses_features = []
        self.nonsilent_ranges = []
        self.spectral_features = []

    def set_silent_ranges(self,min_silence_len=2000,silence_thresh=-40):
        """Attributes a list of all silent sections [start, end] in milliseconds of audio_segment.
        Arguments :
            min_silence_len - the minimum length for any silent section
            silence_thresh - the upper bound for how quiet is silent in dFBS
        """
        self.silent_ranges = pydub.silence.detect_silence(self.audio, min_silence_len=min_silence_len, 
                                                    silence_thresh=silence_thresh, seek_step=20)
    
    def set_pauses_features(self) :    
        """Attributes the min, max and mean of pauses in seconds"""
        silence_durations = [round((e-s)/1000) for (s,e) in self.silent_ranges]

        if len(silence_durations) == 0 :
            self.pauses_features = [0,0,0,0]
        else :
            durations      = [round(d/self.length,6) for d in silence_durations] #normalization
            nb_long_pauses = round(len(durations)/self.length,6)
            mean_pauses    =  np.mean(durations)
            max_pauses     = np.max(durations)
            self.pauses_features = [nb_long_pauses, mean_pauses, max_pauses]

    def set_nonsilent_ranges(self):
        """Attributes a list of all nonsilent sections of an audio."""
        len_seg = self.length*1000 #in ms

        # if there is no silence, the whole thing is nonsilent
        if self.silent_ranges==[]:
            self.nonsilent_ranges = [[0, len_seg]]
            return ;

        # short circuit when the whole audio segment is silent
        if self.silent_ranges[0][0] == 0 and self.silent_ranges[0][1] == len_seg:
            self.nonsilent_ranges = []
            return ;

        prev_end_i = 0
        nonsilent_ranges = []
        for start_i, end_i in self.silent_ranges:
            nonsilent_ranges.append([prev_end_i, start_i])
            prev_end_i = end_i

        if end_i != len_seg:
            nonsilent_ranges.append([prev_end_i, len_seg])

        if nonsilent_ranges[0] == [0, 0]:
            nonsilent_ranges.pop(0)

        self.nonsilent_ranges = nonsilent_ranges

    def split_non_silent(self,keep_silence=1000):
        """Attributes list of audio segments from splitting audio_segment on silent sections"""
        def pairwise(iterable):
            "s -> (s0,s1), (s1,s2), (s2, s3), ..."
            a, b = itertools.tee(iterable)
            next(b, None)
            return zip(a, b)

        if isinstance(keep_silence, bool):
            keep_silence = len(self.audio) if keep_silence else 0

        output_ranges = [
            [ start - keep_silence, end + keep_silence ] for (start,end) in self.nonsilent_ranges
        ]

        for range_i, range_ii in pairwise(output_ranges):
            last_end = range_i[1]
            next_start = range_ii[0]
            if next_start < last_end:
                range_i[1] = (last_end+next_start)//2
                range_ii[0] = range_i[1]

        self.audio = [self.audio[ max(start,0) : min(end,len(self.audio)) ] for start,end in output_ranges]
        self.audio = [item for sublist in self.audio for item in sublist]
        self.length = round(len(self.audio)/1000)

    def preprocessing(self,min_silence_len=2000,silence_thresh=-40,keep_silence=1000):
        # Detecting silent parts
        self.set_silent_ranges(min_silence_len=min_silence_len,silence_thresh=silence_thresh)
        # Calculating pause-related features
        self.set_pauses_features()
        # Getting nonsilent parts
        self.set_nonsilent_ranges()
        # Removing silent parts from audio
        self.split_non_silent(keep_silence=keep_silence)

        # Calculating spectral features
        self.spectral_features = spectral_features(self.audio)


def spectral_features(audio):
    """Extracts spectral features from an audio, ie the means of the following :
    - Spectral centroid,
    - Spectral bandwidth,
    - Spectral rolloff,
    - Zero crossing rate,
    - Mel-Frequency Cepstral Coefficients(MFCCs),
    - Chroma features"""
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