import pydub
import librosa
import os
import numpy as np

def split_questions(video_folder,df_startend,filename):
    df_startend = df_startend[df_startend['mail']==filename.split('.mp4',2)[0]]
    #Read audio
    audio = pydub.AudioSegment.from_file(video_folder+filename,'mp4')
    #split audio
    audios = [audio[s:e] for (s,e) in zip(df_startend['start'],df_startend['end'])]
    return audios

def pydub2librosa(audio):
    audio = audio.set_channels(1) # to mono audio
    y = audio.get_array_of_samples()
    y = librosa.util.buf_to_float(y,n_bytes=2,dtype=np.float32)
    return y, audio.frame_rate