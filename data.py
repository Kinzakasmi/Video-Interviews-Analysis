import pandas as pd
from audio_feats_extract import Audio, pydub2librosa, split_questions
from speech_feats_extract import Lexic

class Interview():
    def __init__(self,audio,email,question,min_silence_len=2000,silence_thresh=-30,keep_silence=1000,
                n_fft=2048,hop_length=512):
        self.Audio = Audio(audio,email,question,min_silence_len,silence_thresh,keep_silence,n_fft,hop_length)
        self.Audio.preprocessing()

        self.Lexic = Lexic(self.Audio.audio,self.Audio.prosodic_features['originaldur'])
        self.Lexic.preprocessing()

        self.features = self.set_features()

    def set_features(self):
        spectral_features          = self.Audio.spectral_features
        spectral_features['stat']  = spectral_features.index
        spectral_features['index'] = 0
        spectral_features          = pd.pivot_table(spectral_features,index='index',columns='stat',values=spectral_features.columns[:-2],aggfunc='first')
        
        prosodic_features = self.Audio.prosodic_features

        lexical_features = self.Lexic.lexical_features

        features = pd.concat([prosodic_features, spectral_features, lexical_features],axis=1)
        features['id']    = self.Audio.email + '_' + str(self.Audio.question)
        features = features.set_index('id')

    def get_features(self):
        return self.features


def read_interview(video_folder,df_startend,filename):
    print(filename)
    #Loading and splitting
    audios = split_questions(video_folder,df_startend,filename)
    #Preprocessing
    interviews = [Interview(audio, filename.split('.mp4',2)[0], i+1) for (i,audio) in enumerate(audios)]
    return interviews

def get_features(interviews):
    feats = []
    for interview in interviews: 
        features = interview.get_features()
        features = features.set_index('id')
        feats.append(features)

    feats = pd.concat(feats,axis=0)
    feats.columns = map(lambda c: '_'.join(c) if isinstance(c, tuple) else c, feats.columns)
    return feats

def get_scores(df_name):
    scores = [pd.read_excel(df_name,i) for i in range(4)]
    scores_all = pd.concat(scores)

    scores_all = scores_all.groupby(level=-1).mean()
    scores_all['id']    = scores[0]['email'] + '_' + scores[0]['question'].astype('str')
    return scores_all

def merge_scores_feats(scores,feats):
    df = feats.copy()
    df = df.merge(scores,on=['id']).set_index('id')

    feats  = df.drop(columns=['Q'+str(i+1) for i in range(21)])
    scores = df[['Q'+str(i+1) for i in range(21)]]

    feats  = feats.drop(columns=['question','bdd','time'])
    return feats, scores